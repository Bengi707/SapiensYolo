# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import os
import random
import cv2
import cv.mmcv
import engine.mmengine as mmengine
import numpy as np
from cv.mmcv.transforms import RandomFlip as MMCV_RandomFlip
from cv.mmcv.transforms import Resize as MMCV_Resize
from cv.mmcv.transforms.base import BaseTransform
from cv.mmcv.transforms.utils import cache_randomness
from engine.mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
from engine.mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from cv.mmcv.transforms import to_tensor
from mmseg.structures import SegDataSample
from engine.mmengine.structures import PixelData

try:
    from libcom import color_transfer
    LIBCOM_INSTALLED = True
except ImportError:
    color_transfer = None
    LIBCOM_INSTALLED = False

try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False

@TRANSFORMS.register_module()
class RandomNormalResizeCompensate(BaseTransform):

    def __init__(self):
        super().__init__()

    ## only to resize the depth and mask according to the image
    def transform(self, results: dict) -> dict:
        assert 'scale' in results.keys() and 'scale_factor' in results.keys()

        target_width, target_height = results['scale']
        keep_ratio = results['keep_ratio']

        # Resize mask to the target size
        mask = results['mask']
        if keep_ratio:
            mask_resized, scale_factor = mmcv.imrescale(
                mask,
                results['scale'],
                interpolation='nearest',
                return_scale=True,
                backend='cv2')
        else:
            mask_resized, w_scale, h_scale = mmcv.imresize(
                mask,
                results['scale'],
                interpolation='nearest',
                return_scale=True,
                backend='cv2')

        results['mask'] = mask_resized
        assert results['mask'].shape[0] == results['img'].shape[0] and results['mask'].shape[1] == results['img'].shape[1]

        ### resize gt_normal and mask same as shape of img
        if 'gt_normal' in results.keys():
            gt_normal = results['gt_normal']

            ## keep_ratio is true from RandomResize in the config
            if keep_ratio:
                gt_normal_resized, scale_factor = mmcv.imrescale(
                    gt_normal,
                    results['scale'],
                    interpolation='nearest',
                    return_scale=True,
                    backend='cv2')
            else:
                gt_normal_resized, w_scale, h_scale = mmcv.imresize(
                    gt_normal,
                    results['scale'],
                    interpolation='nearest',
                    return_scale=True,
                    backend='cv2')

            norms = np.linalg.norm(gt_normal_resized, axis=2, keepdims=True)
            gt_normal_resized = gt_normal_resized / (norms + 1e-6)  # Adding epsilon to avoid division by zero
            results['gt_normal'] = gt_normal_resized

            assert results['gt_normal'].shape[0] == results['img'].shape[0] and results['gt_normal'].shape[1] == results['img'].shape[1]

        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class RandomNormalCrop(BaseTransform):

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]]):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']

        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop depth and crop mask and crop normal
        for key in ['mask', 'gt_normal']:

            if key not in results.keys():
                continue

            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class NormalResize(BaseTransform):
    def __init__(self, scale: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.scale = scale

    def transform(self, results: Dict) -> Dict:

        target_width = self.scale[0]
        target_height = self.scale[1]

        results['img'] = cv2.resize(
                                    results['img'],
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_LINEAR  # Good for image upsampling/downsampling
                                )

        results['mask'] = cv2.resize(
                                    results['mask'],
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_NEAREST
                                )

        if 'gt_normal' in results.keys():
            results['gt_normal'] = cv2.resize(
                                        results['gt_normal'],
                                        (target_width, target_height),
                                        interpolation=cv2.INTER_NEAREST
                                    )

        return results

@TRANSFORMS.register_module()
class NormalRandomFlip(MMCV_RandomFlip):
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip seg map
        results['mask'] = mmcv.imflip(results['mask'], direction=results['flip_direction'])

        if 'gt_normal' in results.keys():
            gt_normal_flipped = mmcv.imflip(results['gt_normal'], direction=results['flip_direction'])

            if results['flip_direction'] == 'horizontal':
                # Negate the X component of the normals for horizontal flip
                # Assuming gt_normal is stored with channel order [X, Y, Z]
                gt_normal_flipped[:, :, 0] = -gt_normal_flipped[:, :, 0]

            results['gt_normal'] = gt_normal_flipped

@TRANSFORMS.register_module()
class GenerateNormalTarget(BaseTransform):
    def __init__(self, background_val=-1000):
        self.background_val = background_val
        return

    def transform(self, results: dict) -> dict:
        gt_normal = results['gt_normal']
        mask = results['mask']
        gt_normal[mask == 0] = self.background_val ## set the background to -1000. the loss uses a threshold of -10 to pick up.
        results['gt_depth_map'] = gt_normal
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class PackNormalInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction',)):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()

        ## is actually normal map
        if 'gt_depth_map' in results:
            gt_depth_map = results['gt_depth_map']
            gt_depth_map = gt_depth_map.transpose(2, 0, 1) ## 3 x H x W, 3 x 1024 x 768
            gt_depth_data = dict(data=to_tensor(gt_depth_map.copy()))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
