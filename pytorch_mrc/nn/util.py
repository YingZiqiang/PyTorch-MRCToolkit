"""
This module implements some useful utility functions for handling the sequence tensor.
"""

import torch
import torch.nn.functional as F

VERY_NEGATIVE_NUMBER = -1e29


def sequence_mask(lengths, maxlen=None, dtype=torch.float32):
    """
    Args:
        lengths: 1D torch.Tensor, shape is `(batch_size)`
    Returns:
        mask: 2D torhc.Tensor, shape is `(batch_size, maxlen)`
    """
    # TODO come from tf.sequence_mask. There should be better implementation
    if maxlen is None:
        maxlen = lengths.max().item()
    mask = torch.zeros(len(lengths), maxlen, device=lengths.device, dtype=dtype)
    for idx, real_len in enumerate(lengths):
        mask[idx, :real_len] = 1
    return mask


def weighted_sum(matrix, attention, dim=1):
    """
    matrix: 3D torch.Tensor
    attention: 2D or 3D torch.Tensor
    dim: which dimension to reduce sum

    Note: In fact, it also supports higher-order tensors, but it needs to keep the dimensions exactly the same.
    """
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(2)
    return (matrix * attention).sum(dim)


def mask_logits(logits, mask):
    """
    logits is a 2D torch.Tensor, its shape usually means (batch_size, max_seq_len).
    mask can be 1D or 2D torch.Tensor. If 1D it means `seq_len`, we will use `sequence_mask` to generate mask,
    if 2D we just use the mask directly.

    Note: In fact, it also supports higher-order tensors, but it needs to keep the dimensions exactly the same.
    """
    if mask.dim() == 1:
        mask = sequence_mask(mask, maxlen=logits.size(1), dtype=torch.float32)
    return logits + (1.0 - mask) * VERY_NEGATIVE_NUMBER


def masked_softmax(logits, mask, dim=-1):
    """
    Firstly, we will do same thing with `mask_logits`, it means logits is a 2D torch.Tensor and mask can be 1D or 2D torch.Tensor.
    Then, we will do `softmax` at selected dimension.

    Note: In fact, it also supports higher-order tensors, but it needs to keep the dimensions exactly the same.
    """
    return F.softmax(mask_logits(logits, mask), dim=dim)


def add_seq_mask(inputs, seq_len, mode='mul', max_len=None):
    mask = sequence_mask(seq_len, maxlen=max_len, dtype=torch.float32).unsqueeze(2)
    if mode == 'mul':
        return inputs * mask
    if mode == 'add':
        mask = (1 - mask) * VERY_NEGATIVE_NUMBER
        return inputs + mask
