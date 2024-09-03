"""
India Driving Dataset (https://idd.insaan.iiit.ac.in/) Loader
For semantic segmentation
Requires dataset to first be converted into .png semantic label format via code from https://github.com/AutoNUE/public-code
(Note: you will need to fix the broken checks for Pillow and install specific libraries)
# E.g. python preperation/createLabels.py  --datadir Datasets/IDD_Segmentation/ --id-type csTrainId --color False --instance False --num-workers 4
Classes are the same as Cityscapes
0:  road
1:  sidewalk
2:  building
3:  wall
4:  fence
5:  pole
6:  traffic light
7:  traffic sign
8:  vegetation
9:  terrain
10: sky
11: person
12: rider
13: car
14: truck
15: bus
16: train
17: motorcycle
18: bicycle
255: void
Michael Smith
McGill University
"""
import os

import datasets.cityscapes_labels as cityscapes_labels
from config import cfg
from datasets import uniform
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder, get_cityscapes_colormap
from runx.logx import logx

class Loader(BaseLoader):
    num_classes = 19
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality='semantic', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.IDD_DIR)
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name
        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:

            self.img_root = os.path.join(self.root, 'leftImg8bit',  mode)
            self.mask_root = os.path.join(self.root, 'gtFine', mode)
            self.all_imgs = self.find_images(self.img_root, self.mask_root)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.root,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    @staticmethod
    def find_images(img_root, mask_root):
        image_mask_pairs = []
        for (dirpath, dirnames, filenames) in os.walk(img_root):
            for filename in filenames:
                if filename.endswith('.png'):
                    image_file = os.path.join(dirpath, filename)
                    drive_no = os.path.basename(dirpath)
                    image_id_num = filename.split('_leftImg8bit.png')[0]
                    mask_file = os.path.join(mask_root, drive_no, image_id_num + '_gtFine_labelcsTrainIds.png')
                    assert os.path.exists(mask_file), f'Missing mask file {mask_file}'
                    image_mask_pairs.append((image_file, mask_file))
        return image_mask_pairs