"""
BDD100K Loader
For semantic segmentation from the BDD100K dataset, which has 10k images for this task (100k images are for other tasks)
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

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder, get_cityscapes_colormap
from datasets import uniform
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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.BDD100K_DIR)
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name
        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            splits = {'train': 'train',
                      'val': 'val',
                      'test': 'test'}
            split_name = splits[mode]
            img_ext = 'jpg'
            mask_ext = 'png'
            self.img_root = os.path.join(self.root, 'images', '10k', split_name)
            self.mask_root = os.path.join(self.root, 'labels', 'sem_seg', 'masks', split_name)
            self.all_imgs = self.find_images(self.img_root, self.mask_root, img_ext,
                                             mask_ext)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.root,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

