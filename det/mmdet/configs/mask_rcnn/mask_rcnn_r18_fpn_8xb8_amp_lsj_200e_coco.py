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
    from .mask_rcnn_r50_fpn_8xb8_amp_lsj_200e_coco import *

from engine.mmengine.model.weight_init import PretrainedInit

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type=PretrainedInit, checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
