# CLAP-eval

Training codebase for Cacophony will be released soon. Data preparation codebase is available at [openSFX-TFShard](https://github.com/gzhu06/openSFX-TFShard).

Evaluation for some CLAP models with audio-text retrieval (zero-shot classification) and Holistic Evaluation of Audio Representations (HEAR) benchmark.

## Requirements

Jax installation is required for the evaluating Cacophony.

## Audio-text retrieval

Evaluation for Cacophony in audio-text retrieval task.
```bash
CUDA_VISIBLE_DEVICES=0 python eval_caco.py
```

## HEAR benchmark

## Acknowledgement
We would like to thank tpu research cloud (TRC) for providing the computational resources for this project.
