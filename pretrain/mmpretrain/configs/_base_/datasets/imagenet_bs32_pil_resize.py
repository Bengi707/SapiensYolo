# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from engine.mmengine.dataset import DefaultSampler

from pretrain.mmpretrain.datasets import (CenterCrop, ImageNet, LoadImageFromFile,
                                 PackInputs, RandomFlip, RandomResizedCrop,
                                 ResizeEdge)
from pretrain.mmpretrain.evaluation import Accuracy

# dataset settings
dataset_type = ImageNet
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=224, backend='pillow'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=ResizeEdge, scale=256, edge='short', backend='pillow'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
)
val_evaluator = dict(type=Accuracy, topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
