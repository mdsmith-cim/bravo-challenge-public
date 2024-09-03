"""
Segment Me If You Can Dataset Loader
Michael Smith, McGill University
Templated off SMIYC_anomaly.py
Ideally this should be done with subclasses rather than copy/paste but I don't think the dataset module loading code would handle that well and I don't have time to figure that out
"""
import os
from pathlib import Path

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder, get_cityscapes_colormap
from datasets.SMIYC_anomaly import Loader as AnomalyLoader
import datasets.cityscapes_labels as cityscapes_labels

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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.SEGMENTMEIFYOUCAN_DIR)
        self.img_root = os.path.join(self.root, 'dataset_ObstacleTrack', 'images')
        self.mask_root = os.path.join(self.root, 'dataset_ObstacleTrack', 'labels_masks')
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name

        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.imgs = make_dataset_folder(eval_folder)
        elif mode == 'val':
            self.imgs = AnomalyLoader.find_images(self.img_root, self.mask_root)
        else:
            raise RuntimeError(f'Unsupported dataset mode {mode}')

        logx.msg('all imgs {}'.format(len(self.imgs)))
