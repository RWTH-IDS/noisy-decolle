# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Sept 5. 2023, T. Stadtmann

import logging
from operator import attrgetter

import torch.distributed as dist
import torch.nn as nn

from ..pq.utils import attrsetter, get_layers
from .modules import ActivationQuantizer, IntConv2d, IntEmbedding, IntLinear
import brevitas.nn as qnn



MAPPING = {nn.Linear: IntLinear, nn.Embedding: IntEmbedding, nn.Conv2d: IntConv2d}
MAPPING_2 = {nn.Linear: qnn.QuantLinear, nn.Conv2d: qnn.QuantConv2d}

def quantize_model_(
    model, p=0.2, bits=8, update_step=1000, method="float", remove_weights=False, percentile=99.99, symmetric = False  #CHANGED histogram to float,added percentile
):
    """
    Replaces all modules with their scalar quantized counterpart and
    registers hooks to quantize the post-ativations of those modules.

    Args:
        - model: a nn.Module
        - p: amount of noise (0 for no noise, 1 to quantize all the weights/activations)
        - bits: number of bits
        - update_step: update quantization parameters every update_step steps
    """
    # quantize all layers
    # remove weights indicates whether the weights extension should be removed, in addition to
    # weight_orig and weight extension on names
    quantized_layers = get_layers(model, "(.*?)", remove_weights=remove_weights)

    for layer in quantized_layers:

        # book-keeping
        is_master_process = (not dist.is_initialized()) or (
            dist.is_initialized() and dist.get_rank() == 0
        )

        # recover module
        module = attrgetter(layer)(model)
        if is_master_process:
            logging.info(
                f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}"
            )

        # quantization params
        q_params = {
            "p": p,
            "update_step": update_step,
            "bits": bits,
            "method": method,
            "counter": 0,
            "percentile": percentile, # Added
        }

        if method == 'brevitas':
            # instantiate the quantized counterpart
            if isinstance(module, nn.Conv2d):
                quantized_module = qnn.QuantConv2d(in_channels=module.in_channels, out_channels=module.out_channels, 
                    kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, 
                    groups=module.groups, bias=True, weight_bit_width = bits )
                quantized_module.weight.data = module.weight.data.clone()
                quantized_module.bias.data = module.bias.data.clone()
            elif isinstance(module, nn.Linear):
                quantized_module = qnn.QuantLinear(in_features=module.in_features, out_features=module.out_features, bias=True, weight_bit_width = bits )
                quantized_module.weight.data = module.weight.data.clone()
                quantized_module.bias.data = module.bias.data.clone()        
            else:
                if is_master_process:
                    logging.info(f"Module {module} not yet supported for quantization")
                continue
        else:
            # quantization params
            q_params = {
                "p": p,
                "update_step": update_step,
                "bits": bits,
                "method": method,
                "counter": 0,
                "percentile": percentile, # Added
                "symmetric": symmetric,
            }
            # instantiate the quantized counterpart
            if isinstance(module, tuple(MAPPING.keys())):
                QuantizedModule = MAPPING[module.__class__]
                quantized_module = QuantizedModule.__new__(QuantizedModule)
                params = module.__dict__
                params.update(q_params)
                quantized_module.__dict__.update(params)

            else:
                if is_master_process:
                    logging.info(f"Module {module} not yet supported for quantization")
                continue


        # activation quantization (CHANGE: removed quantization of the activation function)
        # a_q = ActivationQuantizer(quantized_module, p=0, bits=bits, method=method)

        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_module)

    # return name of quantized layers
    return quantized_layers
