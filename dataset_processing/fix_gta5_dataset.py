"""GTA 5 dataset has some weirdness. We go through the images, identify ones that are messed up, and do some resizing to fix."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='GTA5 fixer.')

parser.add_argument('--gta5_path', type=str,
                    help='Folder for the GTA5 dataset. Expected format: <root> / images, <root> / labels',
                    required=True)

args = parser.parse_args()

gta5_path = Path(args.gta5_path)

# From dataset website
mask_dir = gta5_path / 'labels'
img_dir = gta5_path / 'images'
assert mask_dir.exists(), f'{mask_dir} does not exist.'

# Note: get this split data from the original authors's MATLAB .mat file via scipy which can load MATLAB files.
split_data_matlab_fn = gta5_path / 'split.mat'
assert split_data_matlab_fn.exists(), f'Expecting {split_data_matlab_fn} does not exist.'

# Basic sanity check
all_mask_files = list(mask_dir.glob('*.png'))
all_image_files = list(img_dir.glob('*.png'))
assert len(all_image_files) == len(all_mask_files), f'Number of images {len(all_image_files)} does not match number of masks {len(all_mask_files)}!'

split_data_matlab = loadmat(str(split_data_matlab_fn))
trainIds = split_data_matlab['trainIds'].squeeze()
valIds = split_data_matlab['valIds'].squeeze()
testIds = split_data_matlab['testIds'].squeeze()

# Manually going through the dataset we noticed that these two images are nothing but black/white images even though the labels are goood.
# Code below is to remove them from the split data.
to_remove = [15188, 17705]
print(f'Before removing IDs {to_remove}')
print(f'Train ID: {trainIds.shape}')
print(f'Val ID: {valIds.shape}')
print(f'Test ID: {testIds.shape}')
for i in to_remove:
    trainIds = trainIds[trainIds != i]
    valIds = valIds[valIds != i]
    testIds = testIds[testIds != i]
print('After removing IDs...')
print(f'Train ID: {trainIds.shape}')
print(f'Val ID: {valIds.shape}')
print(f'Test ID: {testIds.shape}')

# Create Numpy version of split data
np.savez(gta5_path / 'split.npz', trainIds=trainIds, valIds=valIds, testIds=testIds)

# Remove offending files from disk
for t in to_remove:
    for p in [mask_dir, img_dir]:
        fn = p / f'{t:05d}.png'
        if fn.exists():
            print(f'Removing {fn}')
            fn.unlink()

# Basically any method expects mask size == image size
# This checks for it as the dataset has some cases where this is not the case
known_sizes = []
for mask in tqdm(list(mask_dir.glob('*.png')), desc='Checking mask and image sizes...', unit='img'):
    mask_img = Image.open(mask)
    mask_size = mask_img.size
    image_fn = Path(str(mask).replace('labels', 'images'))
    image = Image.open(image_fn)
    image_size = image.size
    if image_size != mask_size:
        print(f'Image size {image_size} does not match mask size {mask_size} for {image_fn}')
    match_found = False
    for ks in known_sizes:
        if ks == image_size:
            match_found = True
            break
    if not match_found:
        print(f'Image size {image_size} not in known sizes! Adding.')
        known_sizes.append(image_size)

print(f'Saw image sizes: {known_sizes}')

# Not sure what happened, but most images are (1914, 1052) and some are off by a few pixels. Not sure if scaling or cropping
# but visually it still looks fine so we just resize them all here to be the same
# Being a video game the source should have had consistent resolution
backup_images = gta5_path / 'backup_modified' / 'images'
backup_masks = gta5_path / 'backup_modified' / 'labels'
backup_images.mkdir(exist_ok=True, parents=True)
backup_masks.mkdir(exist_ok=True, parents=True)

desired_size = (1914, 1052)
for mask in tqdm(list(mask_dir.glob('*.png')), unit='img'):
    mask_img = Image.open(mask)
    mask_size = mask_img.size
    image_fn = Path(str(mask).replace('labels', 'images'))
    image = Image.open(image_fn)
    image_size = image.size

    if image_size != desired_size:
        print(f'mage of size {image_size} does not match {desired_size} for {image_fn}')
        backup_fn = backup_images / image_fn.name
        print(f'Making backup of {image_fn} to {backup_fn}')
        shutil.copy2(image_fn, backup_fn)
        new_image = image.resize(desired_size)
        print(f'Saving resized image to {image_fn}')
        new_image.save(image_fn)

    if mask_size != desired_size:
        print(f'Mask of size {mask_size} does not match {desired_size} for {mask}')
        backup_fn = backup_masks / mask.name
        print(f'Making backup of {mask} to {backup_fn}')
        shutil.copy2(mask, backup_fn)
        new_mask = mask_img.resize(desired_size, Image.Resampling.NEAREST)
        print(f'Saving resized mask to {mask}')
        new_mask.save(mask)