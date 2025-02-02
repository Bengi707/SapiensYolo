# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from engine.mmengine.config import read_base

with read_base():
    from .rtmdet_ins_l_8xb32_300e_coco import *

from cv.mmcv.transforms.loading import LoadImageFromFile
from cv.mmcv.transforms.processing import RandomResize
from engine.mmengine.hooks.ema_hook import EMAHook

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import (FilterAnnotations,
                                               LoadAnnotations)
from mmdet.datasets.transforms.transforms import (CachedMixUp, CachedMosaic,
                                                  Pad, RandomCrop, RandomFlip,
                                                  Resize, YOLOXHSVRandomAug)
from mmdet.engine.hooks.pipeline_switch_hook import PipelineSwitchHook
from mmdet.models.layers.ema import ExpMomentumEMA

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa
model.update(
    dict(
        backbone=dict(
            deepen_factor=0.33,
            widen_factor=0.5,
            init_cfg=dict(
                type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
        neck=dict(
            in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
        bbox_head=dict(in_channels=128, feat_channels=128)))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(type=CachedMosaic, img_scale=(640, 640), pad_val=114.0),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=CachedMixUp,
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1, 1)),
    dict(type=PackDetInputs)
]

train_pipeline_stage2 = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type=RandomResize,
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1, 1)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs)
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]
