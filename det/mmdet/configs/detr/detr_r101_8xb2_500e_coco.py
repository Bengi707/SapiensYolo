# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from engine.mmengine.config import read_base
from engine.mmengine.model.weight_init import PretrainedInit

with read_base():
    from .detr_r50_8xb2_500e_coco import *

model.update(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type=PretrainedInit, checkpoint='torchvision://resnet101'))))
