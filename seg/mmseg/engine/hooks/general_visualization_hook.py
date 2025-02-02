# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import warnings
from typing import Optional, Sequence

import cv.mmcv
import engine.mmengine.fileio as fileio
from engine.mmengine.hooks import Hook
from engine.mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from engine.mmengine.model import MMDistributedDataParallel

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import os
import cv2
import tempfile

##-----------------------------------------------------------------------------------------------------------
@HOOKS.register_module()
class GeneralVisualizationHook(Hook):
    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 max_samples = 4,
                 vis_image_width = 512,
                 vis_image_height = 1024,
                 backend_args: Optional[dict] = None):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw

        self.max_samples = max_samples
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height

    def after_train_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """

        # ## check if the rank is 0
        if not runner.rank == 0:
            return

        total_curr_iter = runner.iter

        if total_curr_iter % self.interval != 0:
            return

        inputs = data_batch['inputs'] ## list of images as tensor. BGR images. they are made RGB by model preprocessor
        data_samples = data_batch['data_samples']

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis_data')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(total_curr_iter).zfill(6)

        batch_size = min(self.max_samples, len(inputs))
        inputs = inputs[:batch_size]
        data_samples = data_samples[:batch_size]
        seg_logits = outputs['vis_preds'][:batch_size] ## B x 3 x H x W

        # Check if the model is wrapped with MMDistributedDataParallel
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        data_samples_with_pred = model.postprocess_train_result(seg_logits, data_samples) ##

        seg_logits = seg_logits.cpu().detach().numpy() ## B x 3 x H x W
        seg_logits = seg_logits.transpose((0, 2, 3, 1)) ## B x H x W x 3

        vis_images = []

        for i, (input, data_sample_with_pred, seg_logit) in enumerate(zip(inputs, data_samples_with_pred, seg_logits)):
            image = input.permute(1, 2, 0).cpu().numpy() ## bgr image
            image = np.ascontiguousarray(image.copy())

            gt_albedo = data_sample_with_pred.gt_depth_map.data.numpy() ## 3 x H x W
            gt_albedo = gt_albedo.transpose((1, 2, 0)) ## H x W x 3
            gt_albedo = gt_albedo[:, :, ::-1] ## convert rgb to bgr

            mask = gt_albedo > -100
            vis_gt_albedo = (gt_albedo * 255).astype(np.uint8)
            vis_gt_albedo[~mask] = 100  # grey color

            ### resize pred to the size of image
            pred_albedo = cv2.resize(seg_logit, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

            ## clip the pred to [0, 1]
            pred_albedo = np.clip(pred_albedo, 0, 1)
            pred_albedo = pred_albedo[:, :, ::-1] ## convert rgb to bgr

            vis_pred_albedo = (pred_albedo * 255).astype(np.uint8)

            vis_image = np.concatenate([image, vis_gt_albedo, vis_pred_albedo], axis=1)
            vis_image = cv2.resize(vis_image, (3*self.vis_image_width, self.vis_image_height), interpolation=cv2.INTER_AREA)
            vis_images.append(vis_image)

        grid_image = np.concatenate(vis_images, axis=0)

        # Save the grid image to a file
        grid_out_file = '{}_{}.jpg'.format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)
