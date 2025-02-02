# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from engine.mmengine.config import read_base
from engine.mmengine.optim.scheduler.lr_scheduler import MultiStepLR
from engine.mmengine.runner.loops import EpochBasedTrainLoop

with read_base():
    from .detr_r50_8xb2_150e_coco import *

# learning policy
max_epochs = 500
train_cfg.update(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=10)

param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[334],
        gamma=0.1)
]

# only keep latest 2 checkpoints
default_hooks.update(checkpoint=dict(max_keep_ckpts=2))
