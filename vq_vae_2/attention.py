"""
An implementation of multi-head attention, based off of
https://github.com/unixpickle/xformer
"""

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAttention(nn.Module):
    """
    An attention layer that operates on images.

    Args:
        num_channels: the input image depth.
        num_heads: the number of attention heads.
    """

    def __init__(self, num_channels, num_heads=8):
        super().__init__()
        self.attention = MaskedAttention(num_channels, num_heads=num_heads)

    def forward(self, *images, conds=None):
        """
        Apply masked attention to a batch of images.

        Args:
            images: one or more [N x C x H x W] Tensors.
            conds: ignored. Here for compatibility with
              the PixelCNN aggregator.

        Returns:
            A new list of [N x C x H x W] Tensors.
        """
        results = []
        for image in images:
            batch, num_channels, height, width = image.shape
            result = image.permute(0, 2, 3, 1)
            result = result.view(batch, height * width, num_channels)
            result = self.attention(result)
            result = result.view(batch, height, width, num_channels)
            result = result.permute(0, 3, 1, 2)
            results.append(result + image)
        if len(results) == 1:
            return results[0]
        return tuple(results)


class MaskedAttention(nn.Module):
    """
    An attention layer that operates on sequences of the
    shape [N x T x C], where N is the batch size, T is the
    number of timesteps, and C is the number of channels.

    Args:
        num_channels: the number of channels in the input
          sequences.
        num_heads: the number of attention heads to use.
    """

    def __init__(self, num_channels, num_heads=8):
        super().__init__()

        assert not num_channels % num_heads, 'heads must evenly divide channels'
        self.num_channels = num_channels
        self.num_heads = num_heads

        self.kqv_projection = nn.Linear(num_channels, num_channels * 3)
        self.mix_heads = nn.Linear(num_channels, num_channels)

    def forward(self, sequence):
        """
        Apply masked multi-head attention.

        Args:
            sequence: an [N x T x C] Tensor.

        Returns:
            A new [N x T x C] Tensor.
        """
        projected = self.kqv_projection(sequence)
        kqv = torch.split(projected, self.num_channels, dim=-1)
        keys, queries, values = [self._split_heads(x) for x in kqv]
        logits = torch.bmm(queries, keys.permute(0, 2, 1))
        logits /= math.sqrt(self.num_channels / self.num_heads)
        logits += self._logit_mask(sequence.shape[1])
        weights = F.softmax(logits, dim=-1)
        weighted_sum = torch.bmm(weights, values)
        combined = self._combine_heads(weighted_sum)
        return self.mix_heads(combined)

    def _split_heads(self, batch):
        """
        Split up the channels in a batch into groups, one
        per head.

        Args:
            batch: an [N x T x C] Tensor.

        Returns:
            An [N*H x T x C/H] Tensor.
        """
        batch_size = batch.shape[0]
        num_steps = batch.shape[1]
        split_channels = self.num_channels // self.num_heads
        batch = batch.view(batch_size, num_steps, self.num_heads, split_channels)
        batch = batch.permute(0, 2, 1, 3).contiguous()
        batch = batch.view(batch_size * self.num_heads, num_steps, split_channels)
        return batch

    def _combine_heads(self, batch):
        """
        Perform the inverse of _split_heads().

        Args:
            batch: an [N*H x T x C/H] Tensor.

        Returns:
            An [N x T x C] Tensor.
        """
        batch_size = batch.shape[0] // self.num_heads
        num_steps = batch.shape[1]
        split_channels = self.num_channels // self.num_heads
        batch = batch.view(batch_size, self.num_heads, num_steps, split_channels)
        batch = batch.permute(0, 2, 1, 3).contiguous()
        batch = batch.view(batch_size, num_steps, self.num_channels)
        return batch

    def _logit_mask(self, num_steps):
        row_indices = np.arange(num_steps)[:, None]
        col_indices = np.arange(num_steps)[None]
        upper = (row_indices >= col_indices)
        mask = np.where(upper, 0, -np.inf).astype(np.float32)
        return torch.from_numpy(mask).to(next(self.parameters()).device)
