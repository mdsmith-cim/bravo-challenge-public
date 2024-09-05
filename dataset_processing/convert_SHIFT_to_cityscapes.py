"""
The GTA5 dataset uses Cityscapes labels, but it uses the evaluation IDs which are not useful for training, or in my case for the evaluation on the BRAVO dataset. This script converts the masks to use the trainID labels of cityscapes in the range 0-18.
"""
import argparse
from collections import namedtuple
from pathlib import Path
import os
import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser(description='SHIFT -> Cityscales label converter.')

parser.add_argument('--shift_path', type=str,
                    help='Folder for the SHIFT dataset. Expected format: <root> / discrete / [images,labels] / [train,val] / [front,left_90,right_90...] / <sequence> / <img>',
                    required=True)

args = parser.parse_args()

shift_path = Path(args.shift_path)

def get_cityscapes_colormap():
    """From cityscapes code"""
    palette = [128, 64, 128,
               244, 35, 232,
               70, 70, 70,
               102, 102, 156,
               190, 153, 153,
               153, 153, 153,
               250, 170, 30,
               220, 220, 0,
               107, 142, 35,
               152, 251, 152,
               70, 130, 180,
               220, 20, 60,
               255, 0, 0,
               0, 0, 142,
               0, 0, 70,
               0, 60, 100,
               0, 80, 100,
               0, 0, 230,
               119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    return palette

colormap = get_cityscapes_colormap()

# Below from Cityscapes dataset
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

cityscapes_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

cityscapes_evalID_to_trainID = {label.id: label.trainId for label in cityscapes_labels}

###
# SHIFT Dataset provides the following mapping to Cityscapes eval ID:
###
# ID	Name	Color	Cityscapes equivalent	Cityscapes ignore_in_eval
# 0	    unlabeled	( 0, 0, 0)	0	true
# 1	    building	( 70, 70, 70)	11	false
# 2	    fence	(100, 40, 40)	13	false
# 3	    other	( 55, 90, 80)	0	true
# 4	    pedestrian	(220, 20, 60)	24	false
# 5  	pole	(153, 153, 153)	17	false
# 6	    road line	(157, 234, 50)	7	false
# 7	    road	(128, 64, 128)	7	false
# 8	    sidewalk	(244, 35, 232)	8	false
# 9	    vegetation	(107, 142, 35)	21	false
# 10	vehicle	( 0, 0, 142)	26	false
# 11	wall	(102, 102, 156)	12	false
# 12	traffic sign	(220, 220, 0)	20	false
# 13	sky	( 70, 130, 180)	23	false
# 14	ground	( 81, 0, 81)	6	true
# 15	bridge	(150, 100, 100)	15	true
# 16	rail track	(230, 150, 140)	10	true
# 17	guard rail	(180, 165, 180)	14	true
# 18	traffic light	(250, 170, 30)	19	false
# 19	static	(110, 190, 160)	4	true
# 20	dynamic	(170, 120, 50)	5	true
# 21	water	( 45, 60, 150)	0	true
# 22	terrain	(145, 170, 100)	22	false

shift_id_to_cityscapes_eval_id = {
    0: 0, # unlabeled -> unlabeled
    1: 11, # building -> building
    2: 13, # fence -> fence
    3: 0, # other -> unlabeled
    4: 24, # pedestrian -> person
    5: 17, # pole -> pole
    6: 7, # road line -> road
    7: 7, # road -> road
    8: 8, # sidewalk -> sidewalk
    9: 21, # vegetation -> vegetation
    10: 26, # vehicle -> car
    11: 12, # wall -> wall
    12: 20, # traffic sign -> traffic sign
    13: 23, # sky -> sky
    14: 6, # ground -> ground
    15: 15, # bridge -> bridge
    16: 10, # rail track -> rail track
    17: 14, # guard rail -> guard rail
    18: 19, # traffic light -> traffic light
    19: 4, # static -> static
    20: 5, # dynamic -> dynamic
    21: 0, # water -> unlabeled
    22: 22 # terrain -> terrain
}

shift_id_to_cityscapes_trainID = {shift_id: cityscapes_evalID_to_trainID[cityscapes_eval_id] for shift_id, cityscapes_eval_id in shift_id_to_cityscapes_eval_id.items()}

discrete_dir = shift_path / 'discrete'
assert discrete_dir.exists(), f'Expect {discrete_dir} to exist.'
img_dir =  discrete_dir / 'images'
labels_dir = discrete_dir / 'labels'
assert img_dir.exists(), f'Expect {img_dir} to exist.'
assert labels_dir.exists(), f'Expect {labels_dir} to exist.'

views = ('front', 'left_90', 'right_90')
splits = ('train', 'val')

new_mask_dir = discrete_dir / 'labels_cityscapes'
new_mask_dir.mkdir(exist_ok=True)

print('Finding all mask files...')
all_mask_files = []
for view in views:
    for split in splits:
        mask_dir = labels_dir / split / view
        assert mask_dir.exists(), f'Expect {mask_dir} to exist.'
        all_mask_files.extend(list(mask_dir.glob('*/*.png')))

def convert_mask(mask):
    mask_img = Image.open(mask)
    # IDs are stored in red channel following CARLA.
    mask_img_arr = np.asarray(mask_img)[..., 0]
    new_mask_img_arr = np.full_like(mask_img_arr, fill_value=255)

    for shift_id, cityscapes_train_id in shift_id_to_cityscapes_trainID.items():
        new_mask_img_arr[mask_img_arr == shift_id] = cityscapes_train_id
    # We save as 8 bit PNG, single channel with the ID name, and a colormap for easy visualization following the standards of other datasets e.g. Mapillary
    new_mask_img = Image.fromarray(new_mask_img_arr, mode='P')
    new_mask_img.putpalette(colormap)
    mask_save_fn = new_mask_dir / mask.relative_to(labels_dir)
    mask_save_fn.parent.mkdir(exist_ok=True, parents=True)
    new_mask_img.save(mask_save_fn)

process_map(convert_mask, all_mask_files, desc='Converting masks', unit='img', chunksize=16, max_workers=len(os.sched_getaffinity(0)))
print('Done!')
