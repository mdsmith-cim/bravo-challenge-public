"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
^^^ Original source code

Mapillary Dataset Loader - with Cityscapes Labels
Modified by:
Michael Smith
McGill University
"""
import os
import json

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder
from datasets import uniform
import numpy as np


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

        self.root = os.path.join(cfg.DATASET.DATASET_ROOT, cfg.DATASET.MAPILLARY_DIR)
        config_fn = os.path.join(self.root, 'config_cityscapes.json')
        self.fill_colormap_and_names(config_fn)

        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            splits = {'train': 'training',
                      'val': 'validation',
                      'test': 'testing'}
            split_name = splits[mode]
            img_ext = 'jpg'
            mask_ext = 'png'
            self.img_root = os.path.join(self.root, split_name, 'images')
            self.mask_root = os.path.join(self.root, split_name, 'labels_cityscapes')
            self.all_imgs = self.find_images(self.img_root, self.mask_root, img_ext,
                                             mask_ext)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.root,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    def fill_colormap_and_names(self, config_fn):
        """
        Mapillary code for color map and class names

        Outputs
        -------
        self.trainid_to_name
        self.color_mapping
        """
        with open(config_fn) as config_file:
            config = json.load(config_file)
        config_labels_city = config['cityscapes_labels']

        # calculate label color mapping
        colormap = np.zeros(256 * 3, dtype=np.uint8)
        self.trainid_to_name = {}
        for lab in config_labels_city:
            id_num = lab['id']
            colormap[id_num*3:id_num*3+3] = lab['color']
            name = lab['readable']
            self.trainid_to_name[id_num] = name
        self.color_mapping = colormap.tolist()
