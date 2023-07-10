from abc import ABC, abstractmethod
from typing import Tuple, List
from tqdm import tqdm
import json, os, csv, glob
import pandas as pd
from dataclasses import dataclass

AUDIOTEXT_DATA_PATH = '/storageHDD/ge/audio_sfx_wav/'
# AUDIOTEXT_DATA_PATH = '/storageHDD/ge/audio_sfx_raw/'

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
        existing_audiopaths = glob.glob(f'{self.config.data_dir}/{current_split}/*.wav')
#         import random
#         random.shuffle(existing_audiopaths)

        # load meta json file
        vgg_meta_file = os.path.join(self.config.data_dir, 'vggsound_full.json')
        with open(vgg_meta_file, 'r') as f:
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
        audio_files = glob.glob(f'{self.config.data_dir}/{current_split}/*.wav')
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
    
class Clothov2Processor(DatasetProcessor):
    # clothov2 uses a master cvs for each datasplit instead of paired wav-json
    config = Clothov2Config()
    
    def get_filepaths_and_descriptions(self, current_split=''):
        
        # init output lists
        audio_filepath_list = []
        text_dict = {}
        synthetic_text_dict = {}
        
        # load audio filepaths
        audio_files = glob.glob(f'{self.config.data_dir}/{current_split}/*.wav')
        
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
    