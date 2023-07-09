import csv, json
from tqdm import tqdm
from glob import glob
import os
import numpy as np
import torch
import librosa
from src.laion_clap.hook import CLAP_Module

def load_laionclap(ckpt_path='/storage/ge/ckpts/audio-coca/laion/630k-best.pt',
                   fusion=False):
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
        if len(audio_data) < 48000*5:
            short_files += 1
            continue
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
    print(short_files)

def audio_retrieval():
    pass


        
if __name__ == "__main__":
    
    task = 'zs'

    if task == 'zs':
        # eval 1: ZS classification on VGGSound
        #######################################
        # In classification task: 
        # 1) compute all text embedding
        # 2) rank the top text embeddings on the given audio embedding
        #######################################

        ## dataset config definition
        vggpath = '/storage/ge/vggsound/test'
        vggsound_filelist = glob(os.path.join(vggpath, '**/*.wav'), recursive=True)
        vgg_classes_csv = '/storageHDD/ge/audio_sfx_wav/vggsound/vgg_classes.csv'
        with open(vgg_classes_csv, mode ='r')as file:
            # reading the CSV file
            vgg_classes = [line[0] for line in list(csv.reader(file))]

        class_to_index_map = {v: i for i, v in enumerate(vgg_classes)}
        class_to_index_map[''] = len(vgg_classes) + 1

        all_text_embeddings = compute_all_class_embedding(laionclap, vgg_classes,
                                                          prefix='This is a sound of ')
        zs_classification(laionclap, vggsound_filelist, all_text_embeddings, class_to_index_map)
        
    elif task == 'ar':
    
        # eval 2: (ZS) text to audio retrieval on audiocaps test
        #######################################
        # In retrieval task: 
        # 1) compute all text embedding
        # 2) compute all audio embedding
        # 3a) in text to audio: rank the top audio embeddings on the given text embedding
        # 3b) in audio to text: rank the top text embeddings on the given audio embedding
        #######################################
        
        pass