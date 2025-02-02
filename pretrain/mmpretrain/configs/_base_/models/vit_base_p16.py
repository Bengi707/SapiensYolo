# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from engine.mmengine.model.weight_init import KaimingInit

from pretrain.mmpretrain.models import (ImageClassifier, LabelSmoothLoss,
                               VisionTransformer, VisionTransformerClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=VisionTransformer,
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type=KaimingInit,
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type=VisionTransformerClsHead,
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='classy_vision'),
    ))
