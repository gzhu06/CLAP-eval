import jax
import flax
import jax.numpy as jnp
from src.caco.dataset import DatasetConfig, _dataset_process_map, _tokenize_and_numpy
from caco.caco_eval_utils import load_from_list, VGGSoundProcessor, AudioCapsProcessor, Clothov2Processor, WavText5KProcessor
import tensorflow as tf
from einops import rearrange
import csv
from tqdm import tqdm
from src.caco.load_model import load_caco_ast
from src.caco.dataset import Batch
from retrieval_eval_utils import compute_retrieval_metric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='./ckpt.pt', help='model ckpt path')
parser.add_argument('--task', type=str, default='ar', help='evaluation task name')
args = parser.parse_args()

# load blap globally for test
ckpt_path = args.ckpt_path
caco_model_dict = load_caco_ast(ckpt_path, True)
caco_params = flax.jax_utils.replicate(caco_model_dict['caco_params'], devices=jax.local_devices())
caco_model = caco_model_dict['caco_model']
tokenizer = caco_model_dict['tokenizer']

PyTreeDef = type(jax.tree_util.tree_structure(None))
def _tree_map_batch_devices(x: PyTreeDef) -> PyTreeDef:
    return jax.tree_util.tree_map(
        lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
        x
    )
def get_train_input(
    batch: Batch
) -> PyTreeDef:
    batch = dict(
        audio_patches=batch.audio_patches,
        audio_time_inds=batch.audio_time_inds,
        audio_freq_inds=batch.audio_freq_inds,
        audio_mask=batch.audio_mask,
        text_input_ids=batch.text_input_ids,
        text_mask=batch.text_mask,
    )
    batch = jax.tree_util.tree_map(
        lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
        batch
    )
    return batch

def compute_text_embedding(text_batch, model_params):
    return caco_model.apply(
        {'params': model_params},
        text_input_ids=text_batch['text_input_ids'], 
        text_mask=text_batch['text_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
        method=caco_model.get_text_embedding,
    )

def compute_audio_embedding(audio_batch, model_params):
    return caco_model.apply(
        {'params': model_params},
        audio_patches=audio_batch['audio_patches'],
        audio_time_inds=audio_batch['audio_time_inds'],
        audio_freq_inds=audio_batch['audio_freq_inds'],
        audio_mask=audio_batch['audio_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
        method=caco_model.get_audio_embedding,
    )

def compute_all_class_embedding(class_list, max_text_len, prefix=''):

    all_text_embeddings = []
    t_apply = jax.pmap(compute_text_embedding, axis_name='dp')
    for class_text in tqdm(class_list):
        tokenized = tokenizer([prefix + class_text], 
                              padding='max_length', 
                              truncation=True,
                              max_length=max_text_len, 
                              return_tensors='np')
        text_input_ids, text_mask = tokenized['input_ids'], tokenized['attention_mask']
        text_batch = dict(text_input_ids=text_input_ids,
                          text_mask=text_mask)
        text_batch = jax.tree_util.tree_map(
            lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
            text_batch
        )
        text_embedding = t_apply(text_batch, caco_params)
        all_text_embeddings.append(text_embedding)
    all_text_embeddings = jnp.concatenate(all_text_embeddings, axis=0)
    all_text_embeddings = jnp.squeeze(all_text_embeddings, axis=1)

    return all_text_embeddings

def zs_classification(dataprocessor, datasetconfig, all_text_embeddings):

    filepaths, descriptions, computer_captions = dataprocessor.get_filepaths_and_descriptions(current_split='test')

    dataset_len = len(filepaths)
    a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')

    ks = [1, 5, 10]
    total_correct = {str(k): 0 for k in ks}
    for file_idx in tqdm(range(dataset_len)):
        data_dict = load_from_list(file_idx, filepaths, descriptions, computer_captions)
        d_ = _dataset_process_map(data_dict, [0, 1], datasetconfig)
        d = {}
        for d_item in d_:
            d[d_item] = tf.expand_dims(d_[d_item], axis=0)
        d = _tokenize_and_numpy(d, datasetconfig, tokenizer)
        
        batch = get_train_input(d)
        audio_embedding = a_apply(batch, caco_params)
        target_idx = class_to_index_map[bytes.decode(d_['text'].numpy())]
        targets = _tree_map_batch_devices(jnp.array([target_idx]))
        audio_embedding = jnp.squeeze(audio_embedding, axis=1)
        logits = jnp.exp(caco_params['logit_scale']) * audio_embedding @ all_text_embeddings.T
        indices = jnp.argsort(-logits, axis=-1)
        
        for k in ks:
            n_correct = jnp.sum(jnp.any(targets[..., None] == indices[:, :k], axis=-1))
            total_correct[str(k)] += n_correct

    for k in ks:
        print('top '+str(k)+' accuracy:', total_correct[str(k)]/dataset_len)

def audio_retrieval(dataprocessor, datasetconfig, eval_split='test'):
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)
    a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')
    t_apply = jax.pmap(compute_text_embedding, axis_name='dp')
    
    all_text = []
    all_text_embeddings = []
    all_audio = []
    all_audio_embeddings = []
    gt_audio_text = {}
    gt_text_audio = {}
    
    for file_idx in tqdm(range(dataset_len)):
        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        gt_audio_text[audio_name] = []

        # get text embeddings
        audio_descriptions = descriptions[audio_name]['description']
        for audio_description in audio_descriptions:
            
            # get data info
            data_dict = load_from_list(filepaths[file_idx], audio_description)
            text_str = bytes.decode(data_dict['text'][0].numpy())
            gt_audio_text[audio_name].append(text_str) 
            gt_text_audio[text_str] = audio_name
            all_text.append(text_str)

            # prepare for text embedding
            d_ = _dataset_process_map(data_dict, [0, 1], datasetconfig)
            d = {}
            for d_item in d_:
                d[d_item] = tf.expand_dims(d_[d_item], axis=0)
            d = _tokenize_and_numpy(d, datasetconfig, tokenizer)
            batch = get_train_input(d)

            text_embedding = t_apply(batch, caco_params)
            all_text_embeddings.append(text_embedding)

        # get audio embedding
        audio_embedding = a_apply(batch, caco_params)
        all_audio_embeddings.append(audio_embedding)
        all_audio.append(audio_name)
        
    all_text_embeddings = jnp.concatenate(all_text_embeddings, axis=0)
    all_audio_embeddings = jnp.concatenate(all_audio_embeddings, axis=0)
    
    logits_ar=jnp.squeeze(all_text_embeddings, axis=1) @ jnp.squeeze(all_audio_embeddings.T, axis=1)
    
    # evaluation: audio to text
    print('audio to text retrieval:')
    at_indices = jnp.argsort(jnp.transpose(-logits_ar), axis=-1)
    compute_retrieval_metric(at_indices, all_audio, all_text, gt_audio_text)

    # evaluation: text to audio
    print('text to audio retrieval:')
    ta_indices = jnp.argsort(-logits_ar, axis=-1)
    compute_retrieval_metric(ta_indices, all_text, all_audio, gt_text_audio, 'ta')


        
if __name__ == "__main__":
    
    if args.task == 'zs':
        # eval 1: ZS classification on VGGSound
        #######################################
        # In classification task: 
        # 1) compute all text embedding
        # 2) rank the top text embeddings on the given audio embedding
        #######################################

        audio_seg_time = 10
        total_samples = 16000 * audio_seg_time
        max_patches = (160000 // 160 // 16) * 8
        CommondataConfig = DatasetConfig(batch_size=1,
                                            patches_seq_len=max_patches,
                                            time_patch_size=16,
                                            freq_patch_size=16,
                                            max_text_len=100,
                                            synthetic_prob=0.8)

        ## dataset config definition
        vggsounddataprocessor = VGGSoundProcessor()
        vgg_classes_csv = '/storageHDD/ge/audio_sfx_wav/vggsound/vgg_classes.csv'
        with open(vgg_classes_csv, mode ='r')as file:
            # reading the CSV file
            vgg_classes = [line[0] for line in list(csv.reader(file))]

        class_to_index_map = {v: i for i, v in enumerate(vgg_classes)}
        class_to_index_map[''] = len(vgg_classes) + 1

        all_text_embeddings = compute_all_class_embedding(vgg_classes, 
                                                          CommondataConfig.max_text_len, 
                                                          prefix='This is a sound of ')
        zs_classification(vggsounddataprocessor, CommondataConfig, all_text_embeddings)
        
    elif args.task == 'ar':
    
        # eval 2: (ZS) text to audio retrieval on audiocaps test
        #######################################
        # In retrieval task: 
        # 1) compute all text embedding
        # 2) compute all audio embedding
        # 3a) in text to audio: rank the top audio embeddings on the given text embedding
        # 3b) in audio to text: rank the top text embeddings on the given audio embedding
        #######################################

        audio_seg_time = 30
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples // 160 // 16) * 8
        CommondataConfig = DatasetConfig(batch_size=1,
                                         patches_seq_len=max_patches,
                                         time_patch_size=16,
                                         freq_patch_size=16,
                                         max_text_len=100,
                                         synthetic_prob=0.8)
        
        # wavtext5kprocessor = WavText5KProcessor()
        # audio_retrieval(wavtext5kprocessor, CommondataConfig)
        # exit()

        clothov2processor = Clothov2Processor()
        audio_retrieval(clothov2processor, CommondataConfig, 'evaluation')

        audio_seg_time = 10
        total_samples = 16000 * audio_seg_time
        max_patches = (total_samples // 160 // 16) * 8
        ACdataConfig = DatasetConfig(batch_size=1,
                                     patches_seq_len=max_patches,
                                     time_patch_size=16,
                                     freq_patch_size=16,
                                     max_text_len=100,
                                     synthetic_prob=0.8)
        audiocapsprocessor = AudioCapsProcessor()
        audio_retrieval(audiocapsprocessor, ACdataConfig, 'test')
        
