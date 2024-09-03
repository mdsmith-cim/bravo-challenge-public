import argparse
from pathlib import Path
from PIL import Image
import json
from tqdm.auto import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Mapillary -> Cityscapes label converter.')

parser.add_argument('--mapillary_path', type=str, help='Folder for the Mapillary Datasets. Note: expects v 1.x format.', required=True)

args = parser.parse_args()

mapillary_path = Path(args.mapillary_path)

labeled_splits = ('training', 'validation')

# Read label data from JSON
mapillary_json = mapillary_path / 'config.json'
assert mapillary_json.exists(), f'Expect Mapillary config file {mapillary_json} to exist!'
with open(mapillary_json, 'r') as config_file:
    config = json.load(config_file)
config_labels = config['labels']

# Specify new IDs to name mapping
# Taken from Cityscapes
new_id_to_name = {255: 'void',
                  0: 'road',
                  1: 'sidewalk',
                  2: 'building',
                  3: 'wall',
                  4: 'fence',
                  5: 'pole',
                  6: 'traffic light',
                  7: 'traffic sign',
                  8: 'vegetation',
                  9: 'terrain',
                  10: 'sky',
                  11: 'person',
                  12: 'rider',
                  13: 'car',
                  14: 'truck',
                  15: 'bus',
                  16: 'train',
                  17: 'motorcycle',
                  18: 'bicycle'}

# Custom mapping from Mapillary to Cityscapes
# Based off of The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation (https://arxiv.org/pdf/1904.03215)
# But not quite, took a few liberties where I felt it was more appropriate

old_id_to_new_id_mapping = {
 0: 255, #animal--bird -> void
 1: 255, #animal--ground-animal -> void
 2: 255, #construction--barrier--curb -> void
 3: 4, #construction--barrier--fence -> fence
 4: 255, #construction--barrier--guard-rail -> void
 5: 4, #construction--barrier--other-barrier -> fence
 6: 3, #construction--barrier--wall -> wall
 7: 0, #construction--flat--bike-lane -> road
 8: 0, #construction--flat--crosswalk-plain -> road
 9: 1, #construction--flat--curb-cut' -> sidewalk
 10: 0, #construction--flat--parking -> road
 11: 1, #construction--flat--pedestrian-area -> sidewalk
 12: 255, #construction--flat--rail-track -> void
 13: 0, #construction--flat--road -> road
 14: 255, #construction--flat--service-lane -> void (road shoulders it seems, paved/unpaved)
 15: 1, #construction--flat--sidewalk -> sidewalk
 16: 255, #construction--structure--bridge -> void
 17: 2, #construction--structure--building -> building
 18: 255, #construction--structure--tunnel -> void
 19: 11, #human--person -> person
 20: 12, #human--rider--bicyclist -> rider
 21: 12, #human--rider--motorcyclist -> rider
 22: 12, #human--rider--other-rider -> rider
 23: 0, #marking--crosswalk-zebra -> road
 24: 0, #marking--general -> road
 25: 255, #nature--mountain -> void
 26: 9, #nature--sand -> terrain
 27: 10, #nature--sky -> sky
 28: 9, #nature--snow -> terrain
 29: 9, #nature--terrain -> terrain
 30: 8, #nature--vegetation -> vegetation
 31: 255, #nature--water -> void
 32: 255, #object--banner -> void
 33: 255, #object--bench -> void
 34: 255, #object--bike-rack -> void
 35: 255, #object--billboard -> void
 36: 255, #object--catch-basin -> void
 37: 255, #object--cctv-camera -> void
 38: 255, #object--fire-hydrant -> void
 39: 255, #object--junction-box -> void
 40: 255, #object--mailbox -> void
 41: 255, #object--manhole -> void (usually part of road or sidewalk but can't easily map this)
 42: 255, #object--phone-booth -> void
 43: 255, #object--pothole -> void (can be either road or sidewalk)
 44: 255, #object--street-light -> void
 45: 5, #object--support--pole -> pole
 46: 7, #'object--support--traffic-sign-frame -> traffic sign (e.g. highway sign girder structure)
 47: 5, #object--support--utility-pole -> pole
 48: 6, #object--traffic-light -> traffic light
 49: 255, #object--traffic-sign--back -> void (per cityscapes, traffic signs are only annotated from the front)
 50: 7, #object--traffic-sign--front -> traffic sign
 51: 255, #object--trash-can -> void
 52: 18, #object--vehicle--bicycle -> bicycle
 53: 255, #object--vehicle--boat -> void
 54: 15, #object--vehicle--bus -> bus
 55: 13, #object--vehicle--car -> car
 56: 255, #object--vehicle--caravan -> void
 57: 17, #object--vehicle--motorcycle -> motorcycle
 58: 16, #object--vehicle--on-rails -> train
 59: 255, #object--vehicle--other-vehicle -> void
 60: 255, #object--vehicle--trailer -> void
 61: 14, #object--vehicle--truck -> truck
 62: 255, #object--vehicle--wheeled-slow -> void
 63: 255, #void--car-mount -> void
 64: 255, #void--ego-vehicle -> void
 65: 255 #void--unlabeled -> void
}

# Generate a label name lookup similar to the original format
# Had to add ID field as Cityscapes was sequential (0-65) but Cityscapes is 0-18 + 255
config_labels_cityscapes = []
seen_before = []
for i, old_config_label in enumerate(config_labels):
    mapped_id = old_id_to_new_id_mapping[i]
    print(f'Mapping {i} [{old_config_label["readable"]}] to {mapped_id} [{new_id_to_name[mapped_id]}]')
    if mapped_id in seen_before:
        continue
    # Special overrides for certain cases where we have many -> one mappings and we want to keep the original color
    if mapped_id == 255:
        color = config_labels[65]['color']
    elif mapped_id == 0:
        color = config_labels[13]['color']
    elif mapped_id == 1:
        color = config_labels[15]['color']
    elif mapped_id == 4:
        color = config_labels[3]['color']
    elif mapped_id == 5:
        color = config_labels[45]['color']
    elif mapped_id == 7:
        color = config_labels[50]['color']
    elif mapped_id == 9:
        color = config_labels[29]['color']
    elif mapped_id == 12:
        color = config_labels[20]['color']
    else:
        color = old_config_label['color']
    # Note: this colormap should be an exact match for cityscapes
    config_labels_cityscapes.append(
        {'color': color,
         'id': mapped_id,
         'instances': False,
         'readable': new_id_to_name[mapped_id].capitalize(),
         'name': new_id_to_name[mapped_id],
         'evaluate': True}
    )
    seen_before.append(mapped_id)

# Write to new file
config['cityscapes_labels'] = config_labels_cityscapes
new_config_file = mapillary_path / 'config_cityscapes.json'
print(f'Writing new config file to {new_config_file}')
with open(new_config_file, 'w') as config_file:
    json.dump(config, config_file, indent=4)

# Construct color palette
new_colormap = np.zeros(256 * 3, dtype=np.uint8)
# len = 256 * 3 channels = 768 see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.putpalette
for c in config_labels_cityscapes:
    color = c['color']
    id_num = c['id']
    new_colormap[id_num * 3:id_num * 3 + 3] = color
new_colormap = new_colormap.tolist()

print('Creating new .png labels with Cityscapes labels...')
for split in labeled_splits:
    split_path = mapillary_path / split

    images_dir = split_path / 'images'
    assert images_dir, f'Expect image directory {images_dir} to exist!'
    labels_dir = split_path / 'labels'
    assert labels_dir, f'Expect label directory {labels_dir} to exist!'

    new_labels_dir = split_path / 'labels_cityscapes'
    new_labels_dir.mkdir(exist_ok=True)

    old_labels = list(labels_dir.glob('*.png'))

    for old_label in tqdm(old_labels, desc=f'Converting {split} labels', unit='img'):
        old_label_img = Image.open(old_label)
        old_label_img_arr = np.asarray(old_label_img)
        new_label_img_arr = np.full_like(old_label_img_arr, fill_value=255)

        for old_id, new_id in old_id_to_new_id_mapping.items():
            new_label_img_arr[old_label_img_arr == old_id] = new_id
        new_label_img = Image.fromarray(new_label_img_arr, mode='P')
        new_label_img.putpalette(new_colormap)
        new_label_img.save(new_labels_dir / old_label.name)