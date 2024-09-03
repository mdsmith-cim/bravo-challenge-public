"""
Segment Me If You Can Dataset Loader
Michael Smith, McGill University
Built off template from other NVIDIA-provided datasets and BRAVO challenge provided code (https://github.com/valeoai/bravo_challenge/blob/main/README.md) as well as RbA (https://github.com/NazirNayal8/RbA/tree/main)
"""
import os
from pathlib import Path

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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.SEGMENTMEIFYOUCAN_DIR)
        self.img_root = os.path.join(self.root, 'dataset_AnomalyTrack', 'images')
        self.mask_root = os.path.join(self.root, 'dataset_AnomalyTrack', 'labels_masks')
        self.color_mapping = get_cityscapes_colormap()
        self.trainid_to_name = cityscapes_labels.trainId2name

        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.imgs = make_dataset_folder(eval_folder)
        elif mode == 'val':
            self.imgs = self.find_images(self.img_root, self.mask_root)
        else:
            raise RuntimeError(f'Unsupported dataset mode {mode}')
        logx.msg('all imgs {}'.format(len(self.imgs)))

    @staticmethod
    def find_images(img_root: str, mask_root: str) -> list:
        """
        Find image/mask files and return a list of tuples (img, mask) of them.
        """
        files = []
        img_folder = Path(img_root)
        mask_folder = Path(mask_root)
        for img in img_folder.glob('validation*.*'):
            if img.suffix not in ('.jpg', '.webp'):
                continue
            img_path = img
            mask_path = mask_folder / (img.stem + '_labels_semantic.png')
            assert mask_path.exists(), f'Mask {mask_path} does not exist!'
            files.append((str(img_path), str(mask_path)))
        return files
