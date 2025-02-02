# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pretrain.mmpretrain.models import build_classifier

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepLKNet',
        arch='31B',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

if __name__ == '__main__':
    # model.pop('type')
    model = build_classifier(model)
    model.eval()
    print('------------------- training-time model -------------')
    for i in model.state_dict().keys():
        print(i)
