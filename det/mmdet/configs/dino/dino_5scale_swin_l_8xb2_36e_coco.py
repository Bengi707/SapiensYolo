# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from engine.mmengine.config import read_base
from engine.mmengine.runner.loops import EpochBasedTrainLoop

with read_base():
    from .dino_5scale_swin_l_8xb2_12e_coco import *

max_epochs = 36
train_cfg.update(
    dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1))

param_scheduler[0].update(dict(milestones=[27, 33]))
