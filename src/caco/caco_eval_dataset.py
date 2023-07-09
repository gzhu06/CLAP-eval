from abc import ABC, abstractmethod
from typing import Tuple, List
from tqdm import tqdm
from flax import io
import tensorflow as tf
import tensorflow_io as tfio
import json, os, csv, json
import pandas as pd
from .dataset import _get_mel_spectrogram
from dataclasses import dataclass

AUDIOTEXT_DATA_PATH = '/storageHDD/ge/audio_sfx_wav/'
        
@dataclass
class VGGSoundConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'vggsound'
        
@dataclass
class AudioCapsConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'audiocaps'
        
@dataclass
class Clothov2Config:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'clothov2'

@dataclass
class WavText5KConfig:
    data_dir: str = AUDIOTEXT_DATA_PATH + 'wavtext5k'

@dataclass
class DatasetProcessor(ABC):

    @abstractmethod
    def get_filepaths_and_descriptions(self) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        pass
    
class VGGSoundProcessor(DatasetProcessor):
    # paired wav-json file
    config = VGGSoundConfig()
    
    def get_filepaths_and_descriptions(self, current_split='test'):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        synthetic_text_dict = {}
        
        # load audio filepaths
        existing_audiopaths = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')
#         import random
#         random.shuffle(existing_audiopaths)

        # load meta json file
        vgg_meta_file = os.path.join(self.config.data_dir, 'vggsound_full.json')
        with tf.io.gfile.GFile(vgg_meta_file, 'r') as f:
            vgg_meta_dict = json.load(f)
        
        for audiofile in tqdm(existing_audiopaths[:]):

            # get list of text captions
            audio_name = audiofile.split('/')[-1].split('.wav')[0]
            audio_filepath_list.append(audiofile)
            
            # obtain description item # tags and title+text
            text_captions = {}
            text_captions['description'] = [vgg_meta_dict[audio_name]]
            text_dict[audio_name] = text_captions
        
        return audio_filepath_list, text_dict, synthetic_text_dict
    
class AudioCapsProcessor(DatasetProcessor):
    #  AudioCaps uses a master cvs for each datasplit
    config = AudioCapsConfig()
    
    def get_filepaths_and_descriptions(self, current_split='test'):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        synthetic_text_dict = {}
        
        # load audio filepaths
        audio_files = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')
        with open(os.path.join(self.config.data_dir, 'test.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            meta_info_dict = {}
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                if row[1] not in meta_info_dict:
                    meta_info_dict[row[1]] = [row[-1]]
                else:
                    meta_info_dict[row[1]].append(row[-1])

        # load meta files
        for audio_filepath in tqdm(audio_files[:]):
            
            # load audio filepaths
            audio_name = audio_filepath.split('/')[-1].split('.wav')[0]
            audio_filepath_list.append(audio_filepath)
            
            # get list of text captions
            audio_filename = audio_filepath.split('/')[-1]

            # collecting captions
            text_captions = {}
            text_captions['description'] = meta_info_dict[audio_name]
            text_dict[audio_name] = text_captions
            
            # obtain computer description item
        return audio_filepath_list, text_dict, synthetic_text_dict
    
class WavText5KProcessor(DatasetProcessor):
   # WavText5K uses a master cvs for each datasplit
    config = WavText5KConfig()
    
    def get_filepaths_and_descriptions(self, current_split='full'):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        synthetic_text_dict = {}
        
        # load audio filepaths
        audio_files = io.glob(f'{self.config.data_dir}/**/*.wav')
        meta_info_dict = json.loads(tf.io.read_file(os.path.join(self.config.data_dir, 'meta_info.json')).numpy())

        # load meta files
        for audio_filepath in tqdm(audio_files[:]):
            
            # load audio filepaths
            audio_filepath_list.append(audio_filepath)
            audio_name = audio_filepath.split('/')[-1].split('.wav')[0]
            
            # get list of text captions
            audio_filename = audio_filepath.split('/')[-1]

            # collecting captions
            text_captions = {}
            text_captions['description'] = [meta_info_dict[audio_filename]]
            text_dict[audio_name] = text_captions
            
            # obtain computer description item

        return audio_filepath_list, text_dict, synthetic_text_dict
    
class Clothov2Processor(DatasetProcessor):
    # clothov2 uses a master cvs for each datasplit instead of paired wav-json
    config = Clothov2Config()
    
    def get_filepaths_and_descriptions(self, current_split=''):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        synthetic_text_dict = {}
        
        # load audio filepaths
        audio_files = io.glob(f'{self.config.data_dir}/{current_split}/*.wav')
        
        # load meta files
        for audio_filepath in tqdm(audio_files[:]):
            
            # load audio filepaths
            audio_filepath_list.append(audio_filepath)
            audio_name = audio_filepath.split('/')[-1].split('.wav')[0]
            
            # get list of text captions
            audio_filename = audio_filepath.split('/')[-1]
            split = audio_filepath.split('/')[-2]
            if split != current_split:
                continue
            caption_filename = 'clotho_captions_' + split + '.csv'
            caption_path = os.path.join(self.config.data_dir, caption_filename)

            split_df = pd.read_csv(caption_path)
            data_slice = split_df.loc[split_df['file_name'] == audio_filename]
            
            # collecting captions
            text_captions = {}
            text_captions['description'] = []
            for i in range(5):
                text_captions['description'] += data_slice['caption_'+str(i+1)].tolist()
            text_dict[audio_name] = text_captions
            
            # obtain computer description item

        return audio_filepath_list, text_dict, synthetic_text_dict
    
def load_from_list(file_path, description):
    
    data_dict = {}
    audio_name = file_path.split('/')[-1].split('.wav')[0]
    data_dict['filename'] = audio_name
    audiowav, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
    audio = audiowav[:, 0]
    data_dict['spectrogram'] = compute_mel_spec_audiomae(audio)
    data_dict['text'] = tf.convert_to_tensor([description])
    data_dict['synthetic_text'] = tf.reshape(tf.convert_to_tensor(()), (0, 1))
    return data_dict

def compute_mel_spec_ast(filepath, hop_size=320, num_mel=128,
                         max_load_audio_len=16000*90):
    audio_tensor, _ = tf.audio.decode_wav(tf.io.read_file(filepath))
    mel_spec = _get_mel_spectrogram(tf.squeeze(audio_tensor, axis=1), 
                                    hop_size, num_mel, max_load_audio_len)
    return mel_spec

def compute_mel_spec_audiomae(audio_tensor, 
                              hop_length: int=160,
                              window_length: int=400,
                              num_mels: int=128,
                              scale: float=0.2,
                              bias: float=0.9):
    
    
    spec = tfio.audio.spectrogram(audio_tensor, nfft=512, window=window_length, stride=hop_length)
    mel_spec = tfio.audio.melscale(spec, rate=16000, mels=num_mels, fmin=0, fmax=16000/2)
    mel_spec = tf.math.log(mel_spec+1e-5) * scale + bias
    return mel_spec
    
if __name__ == "__main__":
    
    from dataset import DatasetConfig
    import pandas, csv
    import matplotlib.pyplot as plt
    from collections import Counter
    # dataset definition
    VGGSoundConfig = DatasetConfig(batch_size=1,     
                                   patches_seq_len=254,
                                   time_patch_size=16,
                                   freq_patch_size=16,
                                   max_text_len=77,
                                   synthetic_prob=0.8)
    vgg_classes_csv = '/storageHDD/ge/audio_sfx_wav/vggsound/vgg_classes.csv'
    with open(vgg_classes_csv, mode ='r')as file:
        # reading the CSV file
        vgg_classes = [line[0] for line in list(csv.reader(file))]
    vgg_classes.sort()
    class_to_index_map = {v: i for i, v in enumerate(vgg_classes)}
    
    dataprocessor = VGGSoundProcessor()
    filepaths, descriptions, computer_captions = dataprocessor.get_filepaths_and_descriptions(current_split='train')
    dataset_len = len(filepaths)
    total_labels = []
    for idx in range(dataset_len):
        audiofile = filepaths[idx]
        audio_name = audiofile.split('/')[-1].split('.wav')[0]
        text_des = descriptions[audio_name]['description'][0]
        total_labels.append(class_to_index_map[text_des])
        
    cat_counts = Counter(total_labels)
    plt.hist(total_labels, bins=50)
    plt.savefig('trainvgg.png')
    