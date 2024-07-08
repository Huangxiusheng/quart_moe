# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch


class PackedQuantizedTensor:
    '''Object to store a quantized tensor and its scale.'''

    def __init__(self, quantized_x: torch.Tensor, scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self) -> torch.Size:
        return self.quantized_x.size()

    @property
    def device(self) -> torch.device:
        return self.quantized_x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.quantized_x.dtype


def dequantize(W_ints: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None):
    """
    Reconstruct the (approximate) weight tensor from the quantized weights, scales, and offsets.

    Here, repeat_interleave is used apply the scale and offset accross each group.

    The shape of W_ints is (out_features, in_features)
    The shape of scale is (out_features, in_features // group_size)
    The shape of offset is (out_features, in_features // group_size) (optional)
    """
    # device_org = W_ints.device
    # device_new = W_ints.device
    # if device_new.index < 3:
    #     device_new.index += 1
    # else:
    #     device_new.index = 0

    
    # scale = scale.to(device_new)
    # W_ints = W_ints.to(device_new)
    groupsize = W_ints.shape[-1] // scale.shape[-1]
    flag = 0
    if offset is None:
        offset = 0
        flag = 0
    else:
        # offset = torch.repeat_interleave(offset, groupsize, dim=-1).to(device_new)
        offset = torch.repeat_interleave(offset, groupsize, dim=-1)
        flag= 1
    W = (W_ints - offset) * torch.repeat_interleave(scale, groupsize, dim=-1)
    # W = W.to(device_org)
    # scale = scale.to(device_org)
    # W_ints = W_ints.to(device_org)
    # if flag == 1:
    #     offset = offset.to(device_org)
    return W
