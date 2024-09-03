"""
WildDash Loader
Classes have been converted to be the same as Cityscapes
see dataset_processing/dataset_provided_code/wilddash_scripts for relevant scripts to convert

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
import numpy as np
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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.WILDDASH_DIR)
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name
        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:

            self.img_root = os.path.join(self.root, 'images')
            self.mask_root = os.path.join(self.root, 'semantic_csTrain')
            self.all_imgs = self.find_images(self.root, mode, self.img_root, self.mask_root)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.root,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    @staticmethod
    def find_images(root: str, mode: str, img_root: str, mask_root: str) -> list:
        mode_lookup = {'train': 'training', 'val': 'validation'}
        assert mode in ('train','val','test'), f'Invalid mode: {mode}; only train,val or test allowed'
        with open(os.path.join(root, 'random_split', mode_lookup[mode] + '.txt'), 'r') as f:
            split_ids = f.read().splitlines()

        img_list = []

        for id_num in split_ids:
            img_name = id_num + '.jpg'
            mask_name = id_num + '_labelIds.png'
            img_path = os.path.join(img_root, img_name)
            mask_path = os.path.join(mask_root, mask_name)
            assert os.path.exists(img_path), f'Missing image file {img_name}'
            assert os.path.exists(mask_path), f'Missing mask file {mask_name}'
            img_list.append((img_path, mask_path))

        return img_list
