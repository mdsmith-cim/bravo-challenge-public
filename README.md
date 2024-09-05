# Code for Ensemble approach on the Bravo Challenge - July/August 2024
Primary focus: sample diversity

Submissions made for both Tracks 1 and 2

## Overview

For our submission, we used the following "baseline" models:
1. [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation)
2. [RbA](https://github.com/NazirNayal8/RbA)

The code related to the above two models in our repository has been modified, with a particular focus on dataset handling and writing results to disk to be compatible with out approach.

As detailed in our [technical report](https://papers.cim.mcgill.ca/book/8), our submissions to the BRAVO challenge used different combinations of the above two models trained on different datasets.

The overall process is as follows:

1. Executing models and storing logits on disk (`baselines` directory)
2. Processing the logits into an ensemble (`aggregation` directory)

Please refer to the README files in the respective directories for more details on the individual models and the ensemble approach.

The `bravo_toolkit` directory is a copy from the BRAVO challenge repository, and is needed to convert from our format (`.png` files) to that used by the challenge evaluation server.
The `dataset_processing` directory contains scripts needed to convert some datasets used into more compatible models.

## Required environment
To run the code (including the dataset processing code provided by the respective authors), appropriate python packages need to be installed. Specifically:
* Pillow 
* tqdm
* numpy
* pytorch
* imageio
* numpngw
* pandas
* opencv-python
* detectron2
* timm
* fairscale
* zmq
* scipy
* [panopticapi](https://github.com/cocodataset/panopticapi)
* albumentations
* webp
* easydict
* runx

## Dataset preparation

### Cityscapes

* Download [Cityscapes data](https://www.cityscapes-dataset.com/). You need multiple files:
```
gtFine_trainvaltest.zip
leftImg8bit_trainvaltest.zip
```
* Extract into a folder named `Cityscapes-nvidia` in the folder `DATASET_ROOT`.
### Mapillary
* Download [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas) and extract the files into a folder named `mapillary-vistas-1.2` in the folder `DATASET_ROOT`. Note that v1.2 is the version we use.
* Run the script `convert_mapillary_labels_to_cityscapes.py` in the `dataset_processing` folder to convert the labels to the correct format.

### WildDash 2
* Download `wd_public_v2p0.zip` from the [WildDash 2 website](https://wilddash.cc/) and extract the files into a folder named `WildDash2` in the folder `DATASET_ROOT`.
* In `dataset_processing/dataset_provided_code/wilddash_scripts`, run `remap_coco.py` followed by `pano2sem.py` (adjust the relevant paths as appropriate):
```bash
cd dataset_processing/dataset_provided_code/wilddash_scripts
python remap_coco.py --input <DATASET_ROOT>/WildDash2/panoptic.json --trg_dataset cs --annotation_root <DATASET_ROOT>/WildDash2/panoptic --output <DATASET_ROOT>/WildDash2/panoptic_cs.json 
python pano2sem.py --json_path <DATASET_ROOT>/WildDash2/panoptic_cs.json --outp_dir_sem <DATASET_ROOT>/WildDash2/semantic_cs --label_png_dir <DATASET_ROOT>/WildDash2/panoptic_cs
```
* Then run our script `convert_cs_to_csTrain.py` to convert the Cityscapes-formatted labels to Cityscapes-trainID formatted labels as our code expects.
```bash
python convert_cs_to_csTrain.py --wd_path <DATASET_ROOT>/WildDash2/
```
### BDD100K
* Download `bdd100k_sem_seg_labels_trainval.zip` and `10k_images_{train,val}.zip` from the [BDD100K website](https://dl.cv.ethz.ch/bdd100k/data/). Extract to a folder named `bdd100k` in the folder `DATASET_ROOT`.

### BRAVO
* Download the BRAVO dataset following the links from the [main page](https://github.com/valeoai/bravo_challenge). Extract to a folder named `bravo_1.0`. After extraction, the folder `bravo_1.0` should contain subfolders for each split e.g. `bravo_ACDC`, `bravo_SMIYC` etc.

### India Driving Dataset
* Download the dataset from the [main site](https://idd.insaan.iiit.ac.in/). You need the files `IDD Segmentation (IDD 20k Part I)` and `IDD Segmentation (IDD 20k Part II)`.
* Extract to a folder named `IDD_Segmentation` in the folder `DATASET_ROOT` such that it contains the folders `leftImg8bit` and `gtFine`.
* In `dataset_processing/dataset_provided_code/indiadriving-code`, run `createLabels.py`:
```bash
cd dataset_processing/dataset_provided_code/indiadriving-code
python preperation/createLabels.py  --datadir <DATASET_ROOT>IDD_Segmentation/ --id-type csTrainId
```
### GTA5
* Download the dataset from [the website](https://download.visinf.tu-darmstadt.de/data/from_games/)
* Extract to a folder named `GTA5` in the folder `DATASET_ROOT`.
* Also download the sample code with the provided training/val split and place the `split.mat` file in the `GTA5` folder.
* In the `dataset_processing` folder run `fix_gta5_dataset.py` followed by `convert_gta5_to_cityscapes.py`. The former addresses a number of issues with the datasets, such as some bad data and mismatches between mask and image size. The latter converts it to the appropriate format for our code.

### SHIFT
* Download the SHIFT dataset from [the website](https://www.vis.xyz/shift/). Needed are the RGB images and semantic segmentation masks, for both train and val splits as well as the front camera. We do not use the other views (e.g. left_90) but our code is fully compatible with additional views if desired.
* Note that the dataset authors provide a download script that can be used and will create some of the necessary folder structure:
```bash
python download.py <DATASET_ROOT>/SHIFT --split train,val --view front --group img,semseg --shift discrete --framerate images
```
* Once downloaded, ensure the files are extracted to a folder named `SHIFT` in the folder `DATASET_ROOT` with the following folder structure:
```├── discrete
├── images
│   ├── train
│   │   ├── front
│   │   │   ├── 0003-17fb - *.jpg
│   │   │   ├── 0016-1b62 - *.jpg
│   │   │   ├── ....
│   └── val
│       ├── front
│       │   ├── 007b-4e72 - *.jpg
│       │   ├── 0116-4859 - *.jpg
│       │   ├── ....
├── labels
│   ├── train
│   │   ├── front
│   │   │   ├── 0003-17fb - *.png
│   │   │   ├── 0016-1b62 - *.png
│   │   │   ├── ....
│   └── val
│       ├── front
│       │   ├── 007b-4e72 - *.png
│       │   ├── 0116-4859 - *.png
│       │   ├── ....
```
* Run the `convert_SHIFT_to_cityscapes.py` script in the `dataset_processing` folder:
```bash
python convert_SHIFT_to_cityscapes.py --shift_path <DATASET_ROOT>/SHIFT/ --views front
```

## Generating Ensembles

To generate the ensembles, first run the respective baseline models to generate logits. Then navigate to the `aggregation` directory and run `generate_BRAVO_output.py`. It takes as positional arguments a lsit of folders to use for the ensemble; simply specify the folders with the logits generated from each model as desired. Note that different ways of combining samples are possible, but for all BRAVO submissions only the defaults (mean) were used. Finally, use the `bravo_toolkit` provided encoder to re-encode the submission into a format compatible with the online challenge server (see the [repository](https://github.com/valeoai/bravo_challenge) for details).

For example, this command would generate the results for Ensemble A on Track 1:
```bash
DATASETS=('bravo_ACDC' 'bravo_SMIYC' 'bravo_synobjs' 'bravo_outofcontext' 'bravo_synflare' 'bravo_synrain'); for DATASET in "${DATASETS[@]}";do python generate_BRAVO_output.py HMSA/bravo_ds/cityscapes_sota_model/logits/${DATASET} RbA_logits/logits/${DATASET}/swin_l_1dl --out_path Ensemble_A_submission;done
```
