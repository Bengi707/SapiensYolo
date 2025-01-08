# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configs._base_.datasets.goliath import dataset_info as goliath_info

goliath_info = goliath_info.build()
dataset_info = goliath_info.copy()
dataset_info['dataset_name'] = 'mpii2goliath'