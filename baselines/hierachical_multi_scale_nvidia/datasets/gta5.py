"""
GTA5 Loader
Classes are the same as Cityscapes
19 classes - same as Cityscapes
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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.GTA5_DIR)
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name
        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:

            self.img_root = os.path.join(self.root, 'images')
            self.mask_root = os.path.join(self.root, 'labels_cityscapes')
            self.all_imgs = self.find_images(self.root, mode, self.img_root, self.mask_root)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.root,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    @staticmethod
    def find_images(root: str, mode: str, img_root, mask_root):
        split_info = np.load(os.path.join(root, 'split.npz'))
        id_mapping = {'train': split_info['trainIds'], 'val': split_info['valIds'], 'test': split_info['testIds']}
        selected_ids = id_mapping[mode]
        img_list = []

        for id_num in selected_ids:
            img_name = f'{id_num:05d}.png'
            img_list.append((os.path.join(img_root, img_name), os.path.join(mask_root, img_name)))

        return img_list
