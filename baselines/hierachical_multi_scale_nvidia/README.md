# Implementation of the paper  [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821)
Based on the [original Github repository](https://github.com/NVIDIA/semantic-segmentation).

For the old README, see [README_original.md](README_original.md).

## Notes

* The code was used with torch 2.x
* Dockerfile as provided by original authors was not used
* Runx was not used to run any experiments, but is still used internally in the code for some logging and thus must be installed
* We trained models following the examples set by the various files in the `scripts` folder with some customization. Please see our [technical report](https://papers.cim.mcgill.ca/book/8) for details.
* FP16 was used, but Apex was replaced with native Pytorch implementation

## Setup

* Create asset directory somewhere on disk to store large-ish files
```bash
  > mkdir <large_asset_dir>
```
* Update `__C.DEFAULT_ASSETS_PATH` in `config.py` to point at that directory or specify it on the command line via `--assets_path`.
* Update `__C.DATASET.DATASET_ROOT` in `config.py` to point to a folder containing all datasets or specify it on the command line via `--override_dataset_root`.
* Download pretrained models from the original authors [google drive](https://drive.google.com/open?id=1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) and put into `<large_asset_dir>/seg_weights`
  * If following our naming convention, note that we rename `cityscapes_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth` to `cityscapes_SOTA_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth`.
* Download the models we trained on various datasets from our site [here](https://library.cim.mcgill.ca/data/models/bravo_ensemble_models/) and place them into `<large_asset_dir>/seg_weights`.
* Download and prepare the datasets as described below.

## Download/Prepare Data

Please see the instructions in the main [README.md](../../README.md) for details.

## Running the code - examples

The code can be run via `python train.py <args ...>`. Note that multiple GPU support is built in following pytorch's distributed libraries. By default, batches will be split between GPUs (except batch size 1, which forces only a single GPU).  However, when training it is best to explicitly use a set number of GPUs in distributed mode via `torchrun`.

Note that this GPU behaviour is all inherited from the original code and is common to any similarly-written Pytorch code.

Below are some examples of how to run training or inference. For examples from the original authors, look in the `scripts` folder. Note that the `apex` argument has been removed.
### Inference (evaluating on Cityscapes)
```bash
python train.py --dataset ${DATASET} --cv 0 --result_dir <RESULT_DIR> --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val  --snapshot ASSETS_PATH/seg_weights/cityscapes_SOTA_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth
```
### Training
An example for the GTA5 dataset:
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py --assets_path <ASSETS_PATH> --override_dataset_root <DATASET_ROOT> --dataset gta5 --cv 0 --syncbn --distributed --result_dir <RESULT_DIR> --crop_size "1052,1914" --fp16 --bs_trn 3 --bs_val 3 --poly_exp 2 --lr 1e-2 --supervised_mscale_loss_wt 0.05 --max_epoch 175 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --rmi_loss --class_uniform_tile 1024 --snapshot ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth
```

### Generating the logits from inference on multiple datasets
The small script below can be used to generate and save the logits we use for generating our ensemble across multiple datasets. 
```bash
DATASET=bravo
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/cityscapes_sota_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/cityscapes_SOTA_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/bdd100k_from_mapillary_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/bdd100k_from_mapillary_industrious-chicken-ep125.pth
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/idd_fromcityscapes_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/idd_fromcityscapes_outstanding-turtle_ep118.pth
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/mapillary_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/mapillary_pretrain_from_mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake_best_checkpoint_ep13.pth
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/wilddash_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/wilddash_fromcityscapes_nimble-chihuahua_ep115.pth
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/gta5_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/gta5_frommapillary-ep170.pth
python train.py --dataset ${DATASET} --cv 0 --result_dir HMSA/${DATASET}_ds/SHIFT_model/ --fp16 --bs_val 1 --arch ocrnet.HRNet_Mscale --n_scales "0.5,1.0,2.0" --eval val --dump_logits --snapshot ASSETS_PATH/seg_weights/shift_frommapillary_ep17.pth
```