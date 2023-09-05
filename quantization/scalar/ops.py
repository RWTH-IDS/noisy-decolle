# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Sept 5. 2023, T. Stadtmann

import torch
import numpy as np

try:
    import torch.ao.quantization as quantization
except ImportError:
    import torch.quantization as quantization


def emulate_int(w, bits, method, scale=None, zero_point=None, percentile = 99.99):
    q = globals()[f"emulate_int8_{method}"]
    if method == 'float':
        return q(w, scale=scale, zero_point=zero_point, bits=bits, percentile=percentile)
    else:
        return q(w, scale=scale, zero_point=zero_point, bits=bits)


def quantize(w, scale, zero_point, bits=8, method='not_float'):

    if method == 'float':
        # Followed by updated implementation CHANGE
        x_max = np.power(2, bits - 1) - 1
        x_min = -np.power(2, bits - 1)

        return (
            torch.clamp(torch.round(w/scale), x_min, x_max)
        ) * scale
    else:
        # In the default behavior, max_val = 255.
        max_val = 2**bits - 1
        return (
            torch.clamp(torch.round(w / scale + zero_point), 0, max_val) - zero_point
        ) * scale

# CHANGE : Implemented new emulate quantised method; for compatibility reasons kept int8 prefix although it's fixed point floating number
def emulate_int8_float(w, scale=None, zero_point=None, bits = 8, percentile = 99.99, method = 'float'):
    if scale is None:
        scale = torch.quantile(torch.abs(w), percentile/100).item() / (np.power(2,(bits - 1)) - 1)
        scale = torch.Tensor([scale]).cuda().type_as(w)
        zero_point = torch.zeros(1)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits, method=method), scale, zero_point


def emulate_int8_histogram(w, scale=None, zero_point=None, bits=8):
    if scale is None:
        obs = quantization.observer.HistogramObserver()
        obs.to(device=w.device)
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits), scale, zero_point


def emulate_int8_channel(w, scale=None, zero_point=None, bits=8):
    # Not working
    if scale is None:
        obs = quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        obs.to(device=w.device)
        _ = obs(w)
        scale, zero_point = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits), scale, zero_point


def emulate_int8_tensor(w, scale=None, zero_point=None, bits=8):
    if scale is None:
        obs = quantization.observer.MinMaxObserver(quant_min=0, quant_max=2**bits - 1) #Added quant min and quant max so that it doesn't use 8 bit by default
        obs.to(device=w.device)
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, bits=bits), scale, zero_point
