#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""

import json
import os
import shutil
import time
from pathlib import Path

import click
from tqdm import tqdm

@click.command()
@click.argument("module", type=str)
@click.option(
    "--model-path",
    default=None,
    help="Location of model weights file",
    type=click.Path(exists=True),
)
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Location of tasks to compute embeddings on",
    type=str,
)
@click.option(
    "--task",
    default="all",
    help="Task to run. (Default: all)",
    type=str,
)
@click.option(
    "--embedding-name", default="blap", help="embedding name", type=str
)

@click.option(
    "--embeddings-dir", default="embeddings", help="Location to save task embeddings"
)

@click.option(
    "--batch-size", default=1, help="batch size for computing embeddings", type=int
)
@click.option(
    "--max-audio-len", default=160000, help="max audio length in samples", type=int
)


def runner(
    module: str,
    model_path: str = None,
    tasks_dir: str = "tasks",
    task: str = "tasks",
    embeddings_dir: str = "embeddings",
    embedding_name: str = 'blap',
    batch_size: int = 1,
    max_audio_len: int = 160000
) -> None:
    
    # model loading
    if 'blap' in embedding_name:
        from heareval.embeddings.task_blap_embeddings import Embedding, task_embeddings
    elif 'audiomae' in embedding_name:
        from heareval.embeddings.task_audiomae_embeddings import Embedding, task_embeddings
    elif 'laionclap' in embedding_name:
        from heareval.embeddings.task_laionclap_embeddings import Embedding, task_embeddings
    elif 'fair' in embedding_name:
        from heareval.embeddings.task_fairaudiomae_embeddings import Embedding, task_embeddings
    elif 'msclap' in embedding_name:
        from heareval.embeddings.task_msclap_embeddings import Embedding, task_embeddings
        
    embedding = Embedding(model_path, batch_size=batch_size, audio_max_len=max_audio_len)

    # Check for directory containing the tasks
    tasks_dir_path = Path(tasks_dir)
    embeddings_dir_path = Path(embeddings_dir)
    print(embeddings_dir_path)
    if not tasks_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir_path} exists or specify a folder "
            f"containing HEAR tasks using the argument --tasks-dir"
        )

    if task == "all":
        tasks = list(tasks_dir_path.iterdir())
    else:
        tasks = [tasks_dir_path.joinpath(task)]
        assert os.path.exists(tasks[0]), f"{tasks[0]} does not exist"
    for task_path in tqdm(tasks):

        embed_dir = embeddings_dir_path.joinpath(embedding_name)

        task_name = task_path.name
        embed_task_dir = embed_dir.joinpath(task_name)

        done_embeddings = embed_task_dir.joinpath(".done.embeddings")
        if os.path.exists(done_embeddings):
            continue

        if os.path.exists(embed_task_dir):
            shutil.rmtree(embed_task_dir)

        start = time.time()
        task_embeddings(embedding, task_path, embed_task_dir)

        time_elapsed = time.time() - start
        print(
            f"...computed embeddings in {time_elapsed} sec "
        )
        open(embed_task_dir.joinpath("profile.embeddings.json"), "wt").write(
            json.dumps(
                {
                    "time_elapsed": time_elapsed,
                },
                indent=4,
            )
        )

        # Touch this file to indicate that processing completed successfully
        open(done_embeddings, "wt")


if __name__ == "__main__":
    runner()
