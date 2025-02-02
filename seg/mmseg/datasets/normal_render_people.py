# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

import numpy as np
import os
import cv2
import pickle
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
import os.path as osp
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import random
from matplotlib import pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import engine.mmengine.fileio as fileio

##-----------------------------------------------------------------------
@DATASETS.register_module()
class NormalRenderPeopleDataset(BaseSegDataset):
    """
    """

    def __init__(self,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []

        self.rgb_dir = os.path.join(self.data_root, 'rgb')
        self.normal_dir = os.path.join(self.data_root, 'normal')
        self.mask_dir = os.path.join(self.data_root, 'mask')

        print('\033[92mLoading NormalRenderPeople!\033[0m')

        # Create a set of common file names from all three directories
        rgb_files = {x for x in os.listdir(self.rgb_dir) if x.endswith('.png')}
        mask_files = {x for x in os.listdir(self.mask_dir) if x.endswith('.png')}
        normal_files = {x.replace('.npy', '.png') for x in os.listdir(self.normal_dir) if x.endswith('.npy')}

        # Find the intersection of file names between images, masks, and normals
        common_names = rgb_files & mask_files & normal_files

        # Create data list using the common file names
        data_list = [
                {
                    'rgb_path': os.path.join(self.rgb_dir, name),
                    'mask_path': os.path.join(self.mask_dir, name),
                    'normal_path': os.path.join(self.normal_dir, name.replace('.png', '.npy'))
                }
                for name in common_names
            ]

        print('\033[92mDone! NormalRenderPeople. Loaded total samples: {}\033[0m'.format(len(data_list)))
        return data_list

    def get_data_info(self, idx):
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        img = cv2.imread(data_info['rgb_path']) ## bgr image is default
        normal = np.load(data_info['normal_path']) #
        mask = cv2.imread(data_info['mask_path'])
        mask = mask[:, :, 0] ## 1920 x 1440

        ## check if the normal are normalized
        normal_valid = normal[mask > 0]
        norm_normal_valid = np.linalg.norm(normal_valid, axis=1)

        tolerance = 1e-6

        is_normalized = np.all(norm_normal_valid > 1 - tolerance) & np.all(norm_normal_valid < 1 + tolerance)

        if not is_normalized:
            norms = np.linalg.norm(normal, axis=2, keepdims=True) + 1e-6  # Adding epsilon to avoid division by zero
            normal = normal / norms

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box's bounds
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        data_info = {
            'img': img,
            'img_id': os.path.basename(data_info['rgb_path']),
            'img_path': data_info['rgb_path'],
            'gt_normal': normal,
            'mask': mask,
            'id': idx,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
        }

        return data_info
