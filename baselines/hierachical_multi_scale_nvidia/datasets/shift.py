"""
SHIFT dataset (https://www.vis.xyz/shift/get_started/) loader
For semantic segmentation
Requires dataset to be arranged with folder structure
discrete
    - images
        - train
            - front
                - <sequence>
                    - 00000000_img_front.jpg
                    - 00000010_img_front.jpg
                    ...
            - left_90
                Same as above
            - right_90
                Same as above
        - val
            - front
                Same as above
            - left_90
                Same as above
            - right_90
                Same as above
    - labels
        - train
            - front
                <sequence>
                    - 00000000_semseg_front.png
                    - 00000010_semseg_front.png
                    ...
            - left_90
                Same as above
            - right_90
                Same as above
        - val
            - front
                Same as above
            - left_90
                Same as above
            - right_90
                Same as above

Then run the script convert_SHIFT_to_cityscapes to convert the labels from the format provided to cityscapes train ID. This will create the folder
 "labels_cityscapes" in the same format as the "labels" directory.
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
import glob
from tqdm.contrib.concurrent import process_map
from functools import partial

def get_individual_pair(filename: str, img_root: str, mask_root: str) -> tuple:
    mask_rel_path = os.path.relpath(filename, img_root)
    mask_rel_path = mask_rel_path.replace('_img_', '_semseg_').replace('.jpg', '.png')
    mask_file = os.path.join(mask_root, mask_rel_path)
    assert os.path.exists(mask_file), f'Missing mask file {mask_file}'
    return filename, mask_file

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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.SHIFT_DIR)
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name
        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:

            self.img_root = os.path.join(self.root, 'discrete', 'images', mode)
            self.mask_root = os.path.join(self.root, 'discrete', 'labels_cityscapes', mode)
            self.all_imgs = self.find_images(self.img_root, self.mask_root)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.root,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    @staticmethod
    def find_images(img_root: str, mask_root: str) -> list:
        all_images = glob.glob(os.path.join(img_root, '*/*/*.jpg'))

        image_mask_pairs = process_map(partial(get_individual_pair, mask_root=mask_root, img_root=img_root), all_images, desc='Finding all images and masks', unit='img', chunksize=16,
                    max_workers=len(os.sched_getaffinity(0)))
        return image_mask_pairs