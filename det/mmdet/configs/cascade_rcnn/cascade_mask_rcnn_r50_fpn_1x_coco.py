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
    from .._base_.datasets.coco_instance import *
    from .._base_.default_runtime import *
    from .._base_.models.cascade_mask_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *
