#!/usr/bin/env python3

import json
import os.path
import pickle
import random
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf

import tensorflow_io as tfio
import numpy as np
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import flax
from einops import rearrange
# import wandb
from src.local_load_model import load_audiomae
from src.local_eval_dataset import compute_mel_spec_audiomae
from src.dataset import AudioMAEDatasetConfig, Batch

PyTreeDef = type(jax.tree_util.tree_structure(None))
def get_train_input(
    batch: Batch
) -> PyTreeDef:
    batch = dict(
        audio_patches=batch['audio_patches'],
        audio_time_inds=batch['audio_time_inds'],
        audio_freq_inds=batch['audio_freq_inds'],
        audio_mask=batch['audio_mask'],
    )
    batch = jax.tree_util.tree_map(
        lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
        batch
    )
    return batch

class Embedding:
    """
    A wrapper class to help with loading embedding models and computing embeddings

    Args:
        module_name: the import name for the embedding module
        model_path: location to load the model from
    """

    def __init__(
        self,
        model_path: str = None,
        sample_rate: int = 16000, 
        batch_size: int = 1,
        audio_max_len: int = 160000,
        ):
        CommondataConfig = AudioMAEDatasetConfig(audio_segment_len=audio_max_len)
        dataconfig=CommondataConfig
        self.sample_rate = sample_rate

        # load audiomae globally for test
        audiomae_model_dict = load_audiomae(model_path)
        self.audiomae_params = flax.jax_utils.replicate(audiomae_model_dict['audiomae_params'], 
                                                        devices=jax.local_devices())
        self.audiomae_model = audiomae_model_dict['audiomae_model']
        self.dataconfig = dataconfig
        self.batch_size = batch_size
        self.audio_max_len = audio_max_len
        
        # maximum usable patches
        self.max_patches = (dataconfig.audio_segment_len // dataconfig.spec_hop_length // dataconfig.time_patch_size) * (dataconfig.spec_num_mels // dataconfig.freq_patch_size)
        
        def compute_audio_embedding(audio_batch, model_params):
            return self.audiomae_model.apply(
                {'params': model_params},
                x=audio_batch['audio_patches'],
                time_inds=audio_batch['audio_time_inds'],
                freq_inds=audio_batch['audio_freq_inds'],
                mask=audio_batch['audio_mask'],
                method=self.audiomae_model.__call__,
            )
        
        self.a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')
    
    def get_embedding_from_wav(self, audiowav):
        
        audio = audiowav[:, 0]

        # spectrogram feature coputing
        spectrogram = compute_mel_spec_audiomae(audio, hop_length=self.dataconfig.spec_hop_length,
                                                window_length=self.dataconfig.spec_window_length,
                                                num_mels=self.dataconfig.spec_num_mels,
                                                scale=self.dataconfig.spec_scale,
                                                bias=self.dataconfig.spec_bias)

        # remove residual patches
        spectrogram = spectrogram[:int(tf.shape(spectrogram)[0]//self.dataconfig.time_patch_size*self.dataconfig.time_patch_size)]
        
        # actual used patches
        num_time_patches, num_freq_patches = tf.shape(spectrogram)[0]//self.dataconfig.time_patch_size, tf.shape(spectrogram)[1]//self.dataconfig.freq_patch_size
        total_patches = num_time_patches * num_freq_patches
        
        x = tf.reshape(spectrogram, [num_time_patches, self.dataconfig.time_patch_size, 
                                     num_freq_patches, self.dataconfig.freq_patch_size])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [num_time_patches, num_freq_patches, 
                           self.dataconfig.time_patch_size*self.dataconfig.freq_patch_size])
        x = rearrange(x, 't1 f1 h -> (t1 f1) h')
        
        # random sample if sequence is longer
        if total_patches > self.max_patches:
            keep_inds = list(range(total_patches))
            random.shuffle(keep_inds)
            keep_inds = keep_inds[:self.max_patches]
            keep_inds = tf.sort(keep_inds)

            x = tf.gather(x, indices=keep_inds)
            audio_mask = tf.ones(self.max_patches, dtype=tf.int32)
            time_inds = keep_inds // num_freq_patches
            freq_inds = keep_inds % num_freq_patches
        else:
            audio_mask = tf.cast(tf.range(self.max_patches) < total_patches, tf.int32)
            time_inds = (audio_mask * tf.range(self.max_patches)) // num_freq_patches
            freq_inds = (audio_mask * tf.range(self.max_patches)) % num_freq_patches
            x = tf.pad(x, [[0, self.max_patches - total_patches], [0, 0]], 
                       mode='CONSTANT', constant_values = 0)

#         if tf.shape(audio)[0] > self.dataconfig.audio_segment_len:
#             audio_start_ind = random.randint(0, tf.shape(audio)[0]-self.dataconfig.audio_segment_len+1)
#             audio = audio[audio_start_ind:(audio_start_ind + self.dataconfig.audio_segment_len)]
#         original_time_dim = tf.shape(spectrogram)[0]

        return x, time_inds, freq_inds, audio_mask

    def get_embedding_as_numpy(self, audiofiles, embedding_type='scene') -> np.ndarray:
        
        audio_tensor_list = []
        time_inds_list = []
        freq_inds_list = []
        audiomask_list = []
        for audiofile in audiofiles:
            audio, _ = tf.audio.decode_wav(tf.io.read_file(audiofile))
            
            x, time_inds, freq_inds, audio_mask = self.get_embedding_from_wav(audio)
            audio_tensor_list.append(x)
            time_inds_list.append(time_inds)
            freq_inds_list.append(freq_inds)
            audiomask_list.append(audio_mask)
            
        audio_tensors = tf.stack(audio_tensor_list, axis=0)
        freq_inds_tensors = tf.stack(freq_inds_list, axis=0)
        time_inds_tensors = tf.stack(time_inds_list, axis=0)
        audiomask_tensors = tf.stack(audiomask_list, axis=0)
        
        data_dict =  {'audio_patches': audio_tensors,
                      'audio_time_inds': time_inds_tensors,
                      'audio_freq_inds' : freq_inds_tensors,
                      'audio_mask':audiomask_tensors}
        
        batch = get_train_input(data_dict)
        audio_embeddings = self.a_apply(batch, self.audiomae_params)
        audio_embeddings = jnp.squeeze(audio_embeddings, axis=0)
        
        if embedding_type == 'scene':
            audio_embeddings = jnp.mean(audio_embeddings, axis=1) 
            return audio_embeddings
        elif embedding_type == 'event':
            audio_embeddings = flax.linen.avg_pool(audio_embeddings, 
                                                   window_shape=(8,), 
                                                   strides=(8,), padding='VALID')
            time_stamps = np.linspace(0, self.audio_max_len/self.sample_rate*1000, 
                                      num=audio_embeddings.shape[1])
            time_stamps = np.expand_dims(time_stamps, axis=0)

            return audio_embeddings, time_stamps

def get_dataloader_for_embedding(data: Dict, audio_dir: Path):
    audio_filepath_list = []
    label_dict = {}
    for audio_filename in data:

        audio_filepath_list.append(os.path.join(audio_dir, audio_filename))

        text_captions = {}
        text_captions['description'] = [data[audio_filename]]
        label_dict[audio_filename] = text_captions

    return audio_filepath_list, label_dict

def save_scene_embedding_and_labels(
    embeddings: np.ndarray, labels: List[Dict], filenames: Tuple[str], outdir: Path
):
    assert not np.isnan(embeddings).any()
    assert len(embeddings) == len(filenames)
    assert len(labels) == len(filenames)
    for i, filename in enumerate(filenames):
        out_file = outdir.joinpath(f"{filename}")
        np.save(f"{out_file}.embedding.npy", embeddings[i])
        json.dump(labels[i], open(f"{out_file}.target-labels.json", "w"))

def save_timestamp_embedding_and_labels(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    filename: Tuple[str],
    outdir: Path):
    for i, file in enumerate(filename):
        out_file = outdir.joinpath(f"{file}")
        np.save(f"{out_file}.embedding.npy", embeddings[i])
        assert len(timestamps[i].shape) == 1
        json.dump(timestamps[i].tolist(), open(f"{out_file}.timestamps.json", "w"))
        json.dump(labels[i], open(f"{out_file}.target-labels.json", "w"), indent=4)

def get_labels_for_timestamps(labels: List, timestamps: np.ndarray) -> List:
    
    # convert interval label into timestamp label
    from intervaltree import IntervalTree
    # -> List[List[List[str]]]:
    # -> List[List[str]]:
    # TODO: Is this function redundant?
    # A list of labels present at each timestamp
    timestamp_labels = []

    # NOTE: Make sure dataset events are specified in ms.
    assert len(labels) == len(timestamps)
    
    for i, label in enumerate(labels):
        tree = IntervalTree()
        # Add all events to the label tree
        for event in label:
            # We add 0.0001 so that the end also includes the event
            tree.addi(event["start"], event["end"] + 0.0001, event["label"])

        labels_for_sound = []
        # Update the binary vector of labels with intervals for each timestamp
        for j, t in enumerate(timestamps[i]):
            interval_labels: List[str] = [interval.data for interval in tree[t]]
            labels_for_sound.append(interval_labels)
            # If we want to store the timestamp too
            # labels_for_sound.append([float(t), interval_labels])

        timestamp_labels.append(labels_for_sound)

    assert len(timestamp_labels) == len(timestamps)
    return timestamp_labels

def memmap_embeddings(
    outdir: Path,
    prng: random.Random,
    metadata: Dict,
    split_name: str,
    embed_task_dir: Path,
    split_data: Dict,
):
    """
    Memmap all the embeddings to one file, and pickle all the labels.
    (We assume labels can fit in memory.)
    TODO: This writes things to disk double, we could clean that up after.
    We might also be able to get away with writing to disk only once.
    """
    embedding_files = [outdir.joinpath(f"{f}.embedding.npy") for f in split_data.keys()]
    prng.shuffle(embedding_files)

    # First count the number of embeddings total
    nembeddings = 0
    ndim: int
    for embedding_file in tqdm(embedding_files):
        assert embedding_file.exists()
        emb = np.load(embedding_file).astype(np.float32)
        if metadata["embedding_type"] == "scene":
            assert emb.ndim == 1
            nembeddings += 1
            ndim = emb.shape[0]
            assert emb.dtype == np.float32
        elif metadata["embedding_type"] == "event":
            assert emb.ndim == 2
            nembeddings += emb.shape[0]
            ndim = emb.shape[1]
            assert emb.dtype == np.float32
        else:
            raise ValueError(f"Unknown embedding type: {metadata['embedding_type']}")

    open(
        embed_task_dir.joinpath(f"{split_name}.embedding-dimensions.json"), "wt"
    ).write(json.dumps((nembeddings, ndim)))
    embedding_memmap = np.memmap(
        filename=embed_task_dir.joinpath(f"{split_name}.embeddings.npy"),
        dtype=np.float32,
        mode="w+",
        shape=(nembeddings, ndim),
    )
    idx = 0
    labels = []
    filename_timestamps = []
    for embedding_file in tqdm(embedding_files):
        emb = np.load(embedding_file)
        lbl = json.load(
            open(str(embedding_file).replace("embedding.npy", "target-labels.json"))
        )

        if metadata["embedding_type"] == "scene":
            assert emb.ndim == 1
            embedding_memmap[idx] = emb
            # lbl will be a list of labels, make sure that it has exactly one label
            # for multiclass problems. Will be a list of zero or more for multilabel.
            if metadata["prediction_type"] == "multiclass":
                assert len(lbl) == 1
            elif metadata["prediction_type"] == "multilabel":
                assert isinstance(lbl, list)
            else:
                NotImplementedError(
                    "Only multiclass and multilabel prediction types"
                    f"implemented for scene embeddings. Received {metadata['prediction_type']}"
                )

            labels.append(lbl)
            idx += 1
        elif metadata["embedding_type"] == "event":
            assert emb.ndim == 2
            embedding_memmap[idx : idx + emb.shape[0]] = emb
            assert emb.shape[0] == len(lbl)
            labels += lbl

            timestamps = json.load(
                open(str(embedding_file).replace("embedding.npy", "timestamps.json"))
            )
            slug = str(embedding_file).replace(".embedding.npy", "")
            filename_timestamps += [(slug, timestamp) for timestamp in timestamps]
            assert emb.shape[0] == len(
                timestamps
            ), f"{emb.shape[0]} != {len(timestamps)}"
            assert len(lbl) == len(timestamps), f"{len(lbl)} != {len(timestamps)}"

            idx += emb.shape[0]
        else:
            raise ValueError(f"Unknown embedding type: {metadata['embedding_type']}")

    # Write changes to disk
    embedding_memmap.flush()
    # TODO: Convert labels to indices?
    pickle.dump(
        labels,
        open(
            embed_task_dir.joinpath(f"{split_name}.target-labels.pkl"),
            "wb",
        ),
    )
    if metadata["embedding_type"] == "event":
        assert len(labels) == len(filename_timestamps)
        open(
            embed_task_dir.joinpath(f"{split_name}.filename-timestamps.json"),
            "wt",
        ).write(json.dumps(filename_timestamps, indent=4))


def task_embeddings(
    embedding: Embedding,
    task_path: Path,
    embed_task_dir: Path,
):
    prng = random.Random()
    prng.seed(0)

    metadata_path = task_path.joinpath("task_metadata.json")
    metadata = json.load(metadata_path.open())
    label_vocab_path = task_path.joinpath("labelvocabulary.csv")

    # wandb.init(project="heareval", tags=["embedding", task_name])

    # Copy these two files to the embeddings directory,
    # so we have everything we need in embeddings for doing downstream
    # prediction and evaluation.
    if not os.path.exists(embed_task_dir):
        os.makedirs(embed_task_dir)
    shutil.copy(metadata_path, embed_task_dir)
    shutil.copy(label_vocab_path, embed_task_dir)

    for split in metadata["splits"]:
        print(f"Getting embeddings for split: {split}")

        split_path = task_path.joinpath(f"{split}.json")
        assert split_path.is_file()

        # Copy over the ground truth labels as they may be needed for evaluation
        shutil.copy(split_path, embed_task_dir)

        # Root directory for audio files for this split
        audio_dir = task_path.joinpath(str(embedding.sample_rate), split)

        split_data = json.load(split_path.open())
        audio_filepath_list, label_dict = get_dataloader_for_embedding(split_data, audio_dir)

        outdir = embed_task_dir.joinpath(split)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        total_iter = int(np.ceil(len(audio_filepath_list) / embedding.batch_size))
        for i in tqdm(range(total_iter)):
            
#             try:
#                 audio, _ = tf.audio.decode_wav(tf.io.read_file(audio_filepath_list[i]))
#             except:
#                 print(audio_filepath_list[i])
            
            audio_filepath_sublist = audio_filepath_list[i*embedding.batch_size: (i+1)*embedding.batch_size]
            filenames = [filename.split('/')[-1] for filename in audio_filepath_sublist]
            labels = [split_data[filename.split('/')[-1]] for filename in audio_filepath_sublist]
            if metadata["embedding_type"] == "scene":
                
                embeddings = embedding.get_embedding_as_numpy(audio_filepath_sublist, 
                                                              embedding_type='scene')
                save_scene_embedding_and_labels(embeddings, 
                                                labels, 
                                                filenames, 
                                                outdir)

            elif metadata["embedding_type"] == "event": #TODO
                embeddings, timestamps = embedding.get_embedding_as_numpy(audio_filepath_sublist, 
                                                                          embedding_type='event')
                labels = get_labels_for_timestamps(labels, timestamps)
                assert len(labels) == len(filenames)
                assert len(labels[0]) == len(timestamps[0])
                save_timestamp_embedding_and_labels(
                    embeddings, timestamps, labels, filenames, outdir
                )

            else:
                raise ValueError(
                    f"Unknown embedding type: {metadata['embedding_type']}"
                )

        memmap_embeddings(outdir, prng, metadata, split, embed_task_dir, split_data)
