import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

class BRAVO(Dataset):
    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparam = hparams
        self.return_filepath = hparams.return_filepath
        self.transforms = transforms
        self.split = hparams.dataset_mode
        all_splits = ['bravo_ACDC', 'bravo_SMIYC', 'bravo_outofcontext', 'bravo_synflare', 'bravo_synobjs', 'bravo_synrain']
        all_img_suffix = ['.png', '.jpg', '.png', '.png', '.png',  '.png']
        assert self.split in all_splits, f"split {self.split} not supported"
        split_idx = all_splits.index(self.split)
        self.img_suffix = all_img_suffix[split_idx]
        
        self.root = os.path.join(hparams.dataset_root, 'bravo_1.0')
        self.img_root = os.path.join(self.root, self.split)
        self.mask_root = self.img_root
        self.images = []
        for (dirpath, dirnames, filenames) in os.walk(self.img_root):
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.images.append(os.path.join(dirpath, filename))

        self.labels = [''] * len(self.images)
        self.num_samples = len(self.images)

    def __getitem__(self, index):
        image = self.read_image(self.images[index])
        label = np.zeros_like(image)
        label = label[:, :, 0]
        
        if self.transforms is not None:
            aug = self.transforms(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        if self.return_filepath:
            return image, label.type(torch.LongTensor), self.images[index]
        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples

    
    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img