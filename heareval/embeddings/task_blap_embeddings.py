#!/usr/bin/env python3
"""
Compute the embeddings for every task and store to disk.

Since many tasks might be too large to store in GPU memory (or even
CPU memory), and because Wavenet-like models will be expensive at
inference time, we cache all embeddings to disk.

One benefit of this approach is that since all embeddings are cached
as numpy arrays, the final training code can be pytorch-only,
regardless of whether the embedding model is tensorflow based.

"""
import json
import os.path
import pickle
import random
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
import numpy as np
from intervaltree import IntervalTree
from tqdm.auto import tqdm
import jax
import flax
import numpy as np

# import wandb
from src.local_load_model import load_blap
from src.local_eval_dataset import compute_mel_spec_audiomae
from src.dataset import AudioMAEDatasetConfig, DatasetConfig, _dataset_process_map, _tokenize_and_numpy
from src.local_eval_utils import get_train_input

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
        audio_max_len: int = 160000,
        batch_size: int = 1,
        sample_rate: int = 16000):
        
        dataconfig = AudioMAEDatasetConfig(audio_segment_len=audio_max_len)
        self.dataconfig = dataconfig
        
        self.sample_rate = sample_rate

        # load model
        # load blap globally for test
        blap_model_dict = load_blap(model_path)
        self.blap_params = flax.jax_utils.replicate(blap_model_dict['blap_params'], 
                                                    devices=jax.local_devices())
        self.blap_model = blap_model_dict['blap_model']
        self.tokenizer = blap_model_dict['tokenizer']
        self.audio_max_len = audio_max_len
        
        # maximum usable patches
        max_patches = (dataconfig.audio_segment_len // dataconfig.spec_hop_length // dataconfig.time_patch_size) * (dataconfig.spec_num_mels // dataconfig.freq_patch_size)
        
        self.BLAPdataConfig = DatasetConfig(batch_size=batch_size,
                                            patches_seq_len=max_patches,
                                            time_patch_size=16,
                                            freq_patch_size=16,
                                            max_text_len=77,
                                            synthetic_prob=0.8)

        def compute_audio_embedding(audio_batch, model_params):
            return self.blap_model.apply(
                {'params': model_params},
                audio_patches=audio_batch['audio_patches'],
                audio_time_inds=audio_batch['audio_time_inds'],
                audio_freq_inds=audio_batch['audio_freq_inds'],
                audio_mask=audio_batch['audio_mask'],
                deterministic=True,
                return_hidden_state=False,
                normalize=True,
                method=self.blap_model.get_audio_embedding,
            )
        self.a_apply = jax.pmap(compute_audio_embedding, axis_name='dp')

    def get_scene_embedding_as_numpy(self, audiofile, audiolabel) -> np.ndarray:

        data_dict = {}
        data_dict['filename'] = audiofile
        audiowav, _ = tf.audio.decode_wav(tf.io.read_file(audiofile))
        audio = audiowav[:, 0]
        data_dict['spectrogram'] = compute_mel_spec_audiomae(audio, hop_length=self.dataconfig.spec_hop_length, window_length=self.dataconfig.spec_window_length, num_mels=self.dataconfig.spec_num_mels, scale=self.dataconfig.spec_scale, bias=self.dataconfig.spec_bias)
        data_dict['text'] = tf.convert_to_tensor(['description']) # dummy text
        data_dict['synthetic_text'] = tf.reshape(tf.convert_to_tensor(()), (0, 1))

        d_ = _dataset_process_map(data_dict, [0, 1], self.BLAPdataConfig)
        d = {}
        for d_item in d_:
            d[d_item] = tf.expand_dims(d_[d_item], axis=0)
        d = _tokenize_and_numpy(d, self.BLAPdataConfig, self.tokenizer)
        
        batch = get_train_input(d)
        audio_embedding = self.a_apply(batch, self.blap_params)

        return audio_embedding[0]

    # def get_timestamp_embedding_as_numpy(
    #     self, audio: Union[np.ndarray, torch.Tensor]
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     audio = self.as_tensor(audio)
    #     if self.type == TORCH:
    #         with torch.no_grad():
    #             # flake8: noqa
    #             embeddings, timestamps = self.module.get_timestamp_embeddings(  # type: ignore
    #                 audio,
    #                 self.model,
    #             )
    #             gpu_max_mem.measure()
    #             embeddings = embeddings.detach().cpu().numpy()
    #             timestamps = timestamps.detach().cpu().numpy()
    #             return embeddings, timestamps
    #     elif self.type == TENSORFLOW:
    #         # flake8: noqa
    #         embeddings, timestamps = self.module.get_timestamp_embeddings(  # type: ignore
    #             audio,
    #             self.model,
    #         )
    #         gpu_max_mem.measure()
    #         embeddings = embeddings.numpy()
    #         timestamps = timestamps.numpy()
    #         return embeddings, timestamps
    #     else:
    #         raise NotImplementedError("Unknown type")


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
    outdir: Path,
):
    for i, file in enumerate(filename):
        out_file = outdir.joinpath(f"{file}")
        np.save(f"{out_file}.embedding.npy", embeddings[i])
        assert len(timestamps[i].shape) == 1
        json.dump(timestamps[i].tolist(), open(f"{out_file}.timestamps.json", "w"))
        json.dump(labels[i], open(f"{out_file}.target-labels.json", "w"), indent=4)


def get_labels_for_timestamps(labels: List, timestamps: np.ndarray) -> List:
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

        for filename in tqdm(audio_filepath_list):
            labels = [label_dict[filename.split('/')[-1]]['description'][0]]

            if metadata["embedding_type"] == "scene":
                embeddings = embedding.get_scene_embedding_as_numpy(filename, label_dict[filename.split('/')[-1]])
                save_scene_embedding_and_labels(embeddings, labels, [filename.split('/')[-1]], outdir)

            elif metadata["embedding_type"] == "event": #TODO
                embeddings, timestamps = embedding.get_timestamp_embedding_as_numpy(
                    audios
                )
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
