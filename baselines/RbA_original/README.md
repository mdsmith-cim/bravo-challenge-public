# Implementation of RbA for use with Ensembles
[Paper](https://arxiv.org/pdf/2211.14293.pdf)

Based off the original implementation [on Github](https://github.com/NazirNayal8/RbA/) and some portions of the [modified implementation](https://github.com/valeoai/bravo_challenge/tree/main/baselines/RbA) provided by the BRAVO challenge organizers.

## Pretrained models

Pretrained models provided by the original RbA authors are listed in [RbA Model Zoo](MODEL_ZOO.md). Please download the `Swin-B` and `Swin-L` models listed under the `Cityscapes Inlier Training` heading, and copy to the `ckpts` folder.

## Generating logits
Below is an example of how to dump the logits for both the `Swin-B` and `Swin-L` models on the various BRAVO splits. It assumes that the pretrained models are in the `ckpts` folder.
```bash
python dump_output.py  --batch_size 1 --num_workers 4 --out_path RbA_logits --datasets_folder <DATASET_ROOT> --models_folder ckpts/ --model_mode selective --selected_models swin_l_1dl swin_b_1dl --dataset_mode selective --selected_datasets  bravo_ACDC bravo_SMIYC bravo_outofcontext bravo_synflare bravo_synobjs bravo_synrain --dump_logits
```