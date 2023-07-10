import csv, json
from tqdm import tqdm
from glob import glob
import os
import numpy as np
import torch
import librosa
from src.laion_clap.hook import CLAP_Module
from retrieval_eval_utils import compute_retrieval_metric
from retrieval_eval_dataset import AudioCapsProcessor, Clothov2Processor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='./ckpt.pt', help='model ckpt path')
parser.add_argument('--task', type=str, default='ar', help='evaluation task name')
args = parser.parse_args()

def load_laionclap(ckpt_path=args.ckpt_path, fusion=False):
    model = CLAP_Module(enable_fusion=fusion)
    model.load_ckpt(ckpt_path)
    return model

laionclap = load_laionclap()
laionclap.eval()

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def compute_all_class_embedding(clapmodel, class_list, prefix=''):

    all_text_embeddings = []
    for class_text in tqdm(class_list):
        with torch.no_grad():
            text_embedding = clapmodel.get_text_embedding([prefix + class_text], use_tensor=True)
        all_text_embeddings.append(text_embedding[0].cpu().numpy())

    return all_text_embeddings

def zs_classification(clapmodel, vgg_filelist, all_text_embeddings, class_to_index):
    caption_file = '/storageHDD/ge/audio_sfx_wav/vggsound/vggsound_full.json'
    with open(caption_file, 'r') as f:
        vgg_meta_dict = json.load(f)
    
    total_correct = 0
    total_files = 0
    short_files = 0
    all_text_embeddings = torch.from_numpy(np.array(all_text_embeddings)).cuda()
    for vggfile in tqdm(vgg_filelist):
        
        processed_file = os.path.join('/storageHDD/ge/audio_sfx_wav/vggsound/test', 
                                      vggfile.split('/')[-1])
        if not os.path.exists(processed_file):
            continue

        audio_data, _ = librosa.load(vggfile, sr=48000, mono=True) # sample rate should be 48000
        # if len(audio_data) < 48000*5:
        #     short_files += 1
        #     continue
        audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
        audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
        
        audio_embed = clapmodel.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
        ranking = torch.argsort(audio_embed.clone().detach() @ all_text_embeddings.t(), 
                                descending=True)

        file_gt_label = vgg_meta_dict[vggfile.split('/')[-1].split('.wav')[0]]
        gt_idx = class_to_index[file_gt_label]
        if gt_idx == ranking[0][0].cpu().numpy():
            total_correct += 1
            
        total_files += 1
    
    print('top1:', total_correct/total_files)
    print(short_files, total_files)

def audio_retrieval(dataprocessor, clapmodel, eval_split='test'):
    
    filepaths, descriptions, computer_captions = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)
    
    all_text = []
    all_audio = []
    all_text_embeddings = []
    all_audio_embeddings = []
    gt_audio_text = {}
    gt_text_audio = {}
    
    for file_idx in tqdm(range(dataset_len)):

        audio_name = filepaths[file_idx].split('/')[-1].split('.wav')[0]
        gt_audio_text[audio_name] = []

        # get text embeddings
        audio_descriptions = descriptions[audio_name]['description']
        for audio_description in audio_descriptions:

            gt_audio_text[audio_name].append(audio_description) 
            gt_text_audio[audio_description] = audio_name

            with torch.no_grad():
                text_embed = clapmodel.get_text_embedding([audio_description], use_tensor=True)

            all_text.append(audio_description)
            all_text_embeddings.append(text_embed)

        # get audio embeddings
        audio_data, _ = librosa.load(filepaths[file_idx], sr=48000, mono=True) # sample rate should be 48000
        
        audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
        audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
        
        
        with torch.no_grad():
            audio_embed = clapmodel.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
        all_audio_embeddings.append(audio_embed)

        all_audio.append(audio_name)

    all_text_embeddings = torch.cat(all_text_embeddings, axis=0)
    all_audio_embeddings = torch.cat(all_audio_embeddings, axis=0)
    logits_ar=all_text_embeddings @ all_audio_embeddings.T
    
    # evaluation: audio to text
    print('audio to text retrieval:')
    at_indices = torch.argsort(torch.transpose(-logits_ar, 0, 1), axis=-1)
    compute_retrieval_metric(at_indices, all_audio, all_text, gt_audio_text)

    # evaluation: text to audio
    print('text to audio retrieval:')
    ta_indices = torch.argsort(-logits_ar, axis=-1)
    compute_retrieval_metric(ta_indices, all_text, all_audio, gt_text_audio, 'ta')
        
if __name__ == "__main__":

    if args.task == 'zs':
        # eval 1: ZS classification on VGGSound
        #######################################
        # In classification task: 
        # 1) compute all text embedding
        # 2) rank the top text embeddings on the given audio embedding
        #######################################

        ## dataset config definition
        # vggpath = '/storage/ge/vggsound/test'
        vggpath = '/storageHDD/ge/audio_sfx_raw/vggsound/test'
        vggsound_filelist = glob(os.path.join(vggpath, '**/*.wav'), recursive=True)
        vgg_classes_csv = '/storageHDD/ge/audio_sfx_wav/vggsound/vgg_classes.csv'
        with open(vgg_classes_csv, mode ='r')as file:
            # reading the CSV file
            vgg_classes = [line[0] for line in list(csv.reader(file))]

        class_to_index_map = {v: i for i, v in enumerate(vgg_classes)}

        all_text_embeddings = compute_all_class_embedding(laionclap, vgg_classes,
                                                          prefix='This is a sound of ')
        zs_classification(laionclap, vggsound_filelist, all_text_embeddings, class_to_index_map)
        
    elif args.task == 'ar':
    
        # eval 2: (ZS) text to audio retrieval on audiocaps test
        #######################################
        # In retrieval task: 
        # 1) compute all text embedding
        # 2) compute all audio embedding
        # 3a) in text to audio: rank the top audio embeddings on the given text embedding
        # 3b) in audio to text: rank the top text embeddings on the given audio embedding
        #######################################

        audiocapsprocessor = AudioCapsProcessor()
        audio_retrieval(audiocapsprocessor, laionclap, 'test')
        
        clothov2processor = Clothov2Processor()
        audio_retrieval(clothov2processor, laionclap, 'evaluation')

