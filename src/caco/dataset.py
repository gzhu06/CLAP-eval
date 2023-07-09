import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from flax import struct
tf.config.set_visible_devices([], device_type='GPU')
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable, List, Mapping, Optional, Tuple
from einops import rearrange
import jax
import random

@dataclass
class DatasetLoader(ABC):

    @abstractmethod
    def get_dataset(
        self,
        process_index: int,
        num_parallel_reads: int
    ) -> tf.data.Dataset:
        pass

@struct.dataclass
class DatasetConfig:
    batch_size: int
    patches_seq_len: int 
    time_patch_size: int
    freq_patch_size: int
    max_text_len: int
    synthetic_prob: float
        
@struct.dataclass
class AudioMAEDatasetConfig:
    batch_size: int =1
    audio_segment_len: int = 160000
    time_patch_size: int = 16
    freq_patch_size: int = 16

    spec_hop_length: int = 160
    spec_window_length: int = 400
    spec_num_mels: int = 128
    spec_scale: float = 0.2
    spec_bias: float = 0.9

@struct.dataclass
class DatasetState:
    sample_ind: int
    epoch: int

@struct.dataclass
class Batch:
    audio_patches: np.ndarray
    audio_time_inds: np.ndarray
    audio_freq_inds: np.ndarray
    audio_mask: np.ndarray
    text: np.ndarray
    text_input_ids: np.ndarray
    text_mask: np.ndarray

def get_initial_dataset_state(
    config: DatasetConfig
) -> DatasetState:
    state = DatasetState(
        sample_ind=0,
        epoch=0
    )
    return state

def _get_mel_spectrogram(
    audio_tensor: tf.Tensor,
    hop_size: int,
    num_mel: int,
    max_load_audio_len: int,
) -> tf.Tensor:
    audio_tensor = audio_tensor[..., :max_load_audio_len] # TODO(jd)!! one minute limit for now
    spec = tfio.audio.spectrogram(audio_tensor, nfft=1024, window=800, stride=hop_size,)
    mel_spec = tfio.audio.melscale(spec, rate=16000, mels=num_mel, fmin=0, fmax=16000 // 2)
    mel_spec = (tf.math.log(mel_spec + 1e-5) + 4.5) / 5
    return mel_spec

def _get_mel_spectrogram_audiomae(
    audio: tf.Tensor,
    hop_length: int,
    window_length: int,
    num_mels: int,
    scale: float,
    bias: float
) -> tf.Tensor:
    spec = tfio.audio.spectrogram(audio, nfft=512, window=window_length, stride=hop_length)
    mel_spec = tfio.audio.melscale(spec, rate=16000, mels=num_mels, fmin=0, fmax=16000/2)
    mel_spec = tf.math.log(mel_spec+1e-5) * scale + bias
    return mel_spec

def _dataset_process_map(
    batch: Mapping[str, tf.Tensor], 
    seed: List[int], 
    config: DatasetConfig
) -> Mapping[str, tf.Tensor]:
    # convert a batch of spectrogram and text data

    spectrogram = batch['spectrogram']
    original_time_dim = tf.shape(spectrogram)[0]

#     spectrogram = tf.pad(
#         spectrogram, 
#         [[tf.cast(tf.math.ceil(original_time_dim / config.time_patch_size), tf.int32) * config.time_patch_size - original_time_dim, 0], [0, 0]], 
#         mode='CONSTANT', 
#         constant_values=0
#     )
    
    # remove residual patches
    spectrogram = spectrogram[:int(tf.shape(spectrogram)[0]//config.time_patch_size*config.time_patch_size)]
    
    num_time_patches, num_freq_patches = tf.shape(spectrogram)[0]//config.time_patch_size, tf.shape(spectrogram)[1]//config.freq_patch_size
    full_patch_size = num_time_patches * num_freq_patches

    x = tf.reshape(spectrogram, [num_time_patches, config.time_patch_size, 
                                 num_freq_patches, config.freq_patch_size])
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [num_time_patches, num_freq_patches, 
                       config.time_patch_size*config.freq_patch_size])
    x = rearrange(x, 't1 f1 h -> (t1 f1) h')
        
    # random sample if sequence is longer
    if full_patch_size > config.patches_seq_len:
        keep_inds = list(range(full_patch_size))
        random.shuffle(keep_inds)
        keep_inds = keep_inds[:config.patches_seq_len]
        keep_inds = tf.sort(keep_inds)

        x = tf.gather(x, indices=keep_inds)
        audio_mask = tf.ones(config.patches_seq_len, dtype=tf.int32)
        time_inds = keep_inds // num_freq_patches
        freq_inds = keep_inds % num_freq_patches
    else:
        audio_mask = tf.cast(tf.range(config.patches_seq_len) < full_patch_size, tf.int32)
        time_inds = (audio_mask * tf.range(config.patches_seq_len)) // num_freq_patches
        freq_inds = (audio_mask * tf.range(config.patches_seq_len)) % num_freq_patches
        x = tf.pad(x, [[0, config.patches_seq_len - full_patch_size], [0, 0]], 
                   mode='CONSTANT', constant_values = 0)

    text_index = tf.random.stateless_categorical(
        tf.ones((1, tf.shape(batch['text'])[0]), dtype=tf.float32), 1, 
        seed=tf.random.experimental.stateless_fold_in(seed, 1)
    )[0, 0]

    text = batch['text'][text_index]
    
    if tf.size(batch['synthetic_text']) > 0:

        if tf.random.stateless_categorical(tf.math.log([[1-config.synthetic_prob, config.synthetic_prob]]), 1, 
                seed=tf.random.experimental.stateless_fold_in(seed, 2)) > 0:
        
            synthetic_text_index = tf.random.stateless_categorical(
                tf.ones((1, tf.shape(batch['synthetic_text'])[0]), dtype=tf.float32), 1, 
                seed=tf.random.experimental.stateless_fold_in(seed, 1)
            )[0, 0]

            text = batch['synthetic_text'][synthetic_text_index]

    d =  {
        'audio_patches': x,
        'audio_time_inds': time_inds,
        'audio_freq_inds' : freq_inds,
        'audio_mask': audio_mask,
        'text': text,
    }

    if 'data_mask' in batch.keys():
        d['data_mask'] = batch['data_mask']

    return d


def _tokenize_and_numpy(batch, config, tokenize_fn):
    text = [s.decode('utf-8') for s in batch['text']._numpy()]
    tokenize_output = tokenize_fn(text, padding='max_length', truncation=True, max_length=config.max_text_len, return_tensors='np')
    text_input_ids = tokenize_output['input_ids']
    text_mask = tokenize_output['attention_mask']
    return Batch(
        audio_patches=batch['audio_patches']._numpy(),
        audio_time_inds=batch['audio_time_inds']._numpy(),
        audio_freq_inds=batch['audio_freq_inds']._numpy(),
        audio_mask=batch['audio_mask']._numpy(),
        text=text,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
    )

class Dataset:

    def __init__(
        self,
        dataset_loaders: Iterable[DatasetLoader],
        spec_hop_size: int,
        spec_num_mel: int,
        num_parallel_reads: int = 16,
        max_load_audio_len: Optional[int] = None,
        split_tfrecord_shards: bool = True
    ):

        self.spec_hop_size = spec_hop_size
        self.spec_num_mel = spec_num_mel

        all_data = []
        for dataset_loader in dataset_loaders:
            dataset = dataset_loader.get_dataset(jax.process_index(), jax.process_count(), 
                num_parallel_reads, split_tfrecord_shards)
            
            if max_load_audio_len is not None:
                def _audio_crop_map(x):
                    x.update({'audio': x['audio'][:max_load_audio_len]})
                    return x

                dataset = dataset.map(_audio_crop_map)
            
            dataset = dataset.map(lambda x: dict(spectrogram=_get_mel_spectrogram(x['audio'], spec_hop_size, spec_num_mel, max_load_audio_len), **x))
            for i, data in enumerate(dataset):

                if split_tfrecord_shards or (i + jax.process_index()) % jax.process_count() == 0:

                    all_data.append(data)

        self.data = all_data

    def get_train_dataset_generator(
        self,
        config: DatasetConfig,
        tokenize_fn: Callable,
        rng_seed: int,
        resume_state: Optional[DatasetState] = None
    ) -> Generator[Tuple[Batch, DatasetState], None, None]:

        if resume_state is None:
            resume_state = get_initial_dataset_state(config)

        def _get_full_dataset(epoch: int, skip: int = 0) -> tf.data.Dataset:
            d_len = len(self.data)
            ds_seed = epoch * jax.process_count() + jax.process_index() + rng_seed
            ds_seed_stateless = [ds_seed * 2, ds_seed * 2 + 1]
            dataset = tf.data.Dataset.range(d_len)
            dataset = dataset.shuffle(buffer_size=d_len, seed=ds_seed)
            def dmap_data(i):                    
                return self.data[i]['spectrogram'], self.data[i]['text'], self.data[i]['synthetic_text']
            dataset = dataset.map(lambda x: tf.py_function(dmap_data, [x], [tf.float32, tf.string, tf.string]))
            dataset = dataset.map(lambda *x: {'spectrogram': x[0], 'text': x[1], 'synthetic_text': x[2]})

            dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.range(int(4e9)).map(
                lambda x: tf.random.experimental.stateless_fold_in(ds_seed_stateless, x))))
            dataset = dataset.map(lambda d, s: _dataset_process_map(d, s, config))
            dataset = dataset.batch(config.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(2)
            dataset = dataset.skip(skip)
            return dataset

        epoch = resume_state.epoch
        sample_ind = resume_state.sample_ind

        while True:
            dataset = iter(_get_full_dataset(epoch, skip=sample_ind))
            while True:
                try:
                    data = next(dataset)
                    batch = _tokenize_and_numpy(data, config, tokenize_fn=tokenize_fn)
                    sample_ind += 1
                    yield batch, DatasetState(
                        sample_ind=sample_ind,
                        epoch=epoch
                    )
                except StopIteration:
                    break

            sample_ind = 0
            epoch += 1


    def get_eval_dataset_generator(
        self,
        config: DatasetConfig,
        tokenize_fn: Callable,
        rng_seed: int,
        shuffle: bool = False
    ) -> Generator[Tuple[Batch, DatasetState], None, None]:

        def _get_full_dataset(epoch: int, skip: int = 0) -> tf.data.Dataset:
            filler_element = {
                'data_mask': [[False]], 
                'spectrogram': tf.zeros((1, 256, self.spec_num_mel)), # TODO(jd)
                'text': tf.convert_to_tensor([['']]), 
                'synthetic_text': tf.convert_to_tensor([['']])
            }
            filler_dataset = tf.data.Dataset.from_tensor_slices(filler_element)
            d_len = len(self.data)
            ds_seed = epoch * jax.process_count() + jax.process_index() + rng_seed
            ds_seed_stateless = [ds_seed * 2, ds_seed * 2 + 1]
            dataset = tf.data.Dataset.range(d_len)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=d_len, seed=ds_seed)
            def dmap_data(i):                    
                return self.data[i]['spectrogram'], self.data[i]['text'], self.data[i]['synthetic_text']
            dataset = dataset.map(lambda x: tf.py_function(dmap_data, [x], [tf.float32, tf.string, tf.string]))
            dataset = dataset.map(lambda *x: {'spectrogram': x[0], 'text': x[1], 'synthetic_text': x[2]})
            dataset = dataset.map(
                lambda features: dict(data_mask=[True], **features))
            dataset = dataset.concatenate(filler_dataset.repeat(None))

            dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.range(int(4e9)).map(
                lambda x: tf.random.experimental.stateless_fold_in(ds_seed_stateless, x))))
            dataset = dataset.map(lambda d, s: _dataset_process_map(d, s, config))
            dataset = dataset.batch(config.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(2)
            dataset = dataset.skip(skip)
            return dataset

        dataset = iter(_get_full_dataset(0, 0))

        while True:
            data = next(dataset)
            batch = _tokenize_and_numpy(data, config, tokenize_fn=tokenize_fn)
            yield batch, data['data_mask']._numpy()[:, 0]
