# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from engine.mmengine.dist.utils import get_dist_info
from engine.mmengine.optim import DefaultOptimWrapperConstructor
from engine.mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


## for fsdp wrapped model
def fsdp_get_num_layer_for_vit(var_name, num_max_layer):
    ## remove fsdp prefix. eg: backbone._fsdp_wrapped_module.layers.0.ln1.weight -> backbone.layers.0.ln1.weight
    var_name = var_name.replace('_fsdp_wrapped_module.', '') 

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1

def get_num_layer_for_vit(var_name, num_max_layer):
    if '_fsdp_wrapped_module' in var_name:
        return fsdp_get_num_layer_for_vit(var_name, num_max_layer)

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg=None)
        self.layer_decay_rate = paramwise_cfg.get('layer_decay_rate', 0.5)

        super().__init__(optim_wrapper_cfg, paramwise_cfg)

    def add_params(self, params, module, prefix='', lr=None):
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (len(param.shape) == 1 or name.endswith('.bias')
                    or 'pos_embed' in name):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            ## ----------debug--------------            
            # import torch.distributed as dist; import ipdb; 
            # if dist.get_rank() == 0:  # debug only on rank 0
            #     # ipdb.set_trace()
            #     print(f"{name} {param.shape} {param.dtype} layer_id:{layer_id} lr: {scale * self.base_lr}")
            # --------------------------------

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
        params.extend(parameter_groups.values())
