"""

Bravo Challenge Dataset Loader
Michael Smith, McGill University
Built off template from other NVIDIA-provided datasets and BRAVO challenge provided code (https://github.com/valeoai/bravo_challenge/blob/main/README.md)
"""
import os

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder, get_cityscapes_colormap
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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.BRAVO_DIR)
        self.img_root = self.root
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name

        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.imgs = make_dataset_folder(eval_folder)
        elif mode == 'val':
            img_extensions = ('png', 'jpg')
            self.imgs = self.find_images(img_extensions)
        else:
            raise RuntimeError(f'Unsupported dataset mode {mode}')

        logx.msg('all imgs {}'.format(len(self.imgs)))

    def find_images(self, img_extensions):
        """
        Find image files and return a list of tuples (img, empty string) of them, with the empty string taking the place of the mask.
        """
        images = []
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            for filename in filenames:
                if filename.endswith(img_extensions):
                    images.append((os.path.join(dirpath, filename), ''))
        return images
