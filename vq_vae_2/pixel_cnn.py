"""
An implementation of the Gated PixelCNN from
https://arxiv.org/abs/1606.05328.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelCNN(nn.Module):
    """
    A PixelCNN is a stack of PixelConv layers.
    """

    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module('layer_%d' % i, layer)
        self.layers = layers

    def forward(self, images, conds=None):
        """
        Apply the stack of PixelConv layers.

        It is assumed that the first layer is a
        PixelConvA, and the rest are PixelConvB's.
        This way, the first layer takes one input and the
        rest take two.

        Returns:
            A tuple (vertical, horizontal), one for each
              of the two directional stacks.
        """
        outputs = self.layers[0](images, conds=conds)
        for layer in self.layers[1:]:
            outputs = layer(*outputs, conds=conds)
        return outputs


class PixelConv(nn.Module):
    """
    An abstract base class for PixelCNN layers.
    """

    def __init__(self, depth_in, depth_out, cond_depth=None, horizontal=2, vertical=2):
        super().__init__()
        self.depth_in = depth_in
        self.depth_out = depth_out
        self.horizontal = horizontal
        self.vertical = vertical

        self._init_directional_convs()
        self.vert_to_horiz = nn.Conv2d(depth_out * 2, depth_out * 2, 1)
        self.cond_layer = None
        if cond_depth is not None:
            self.cond_layer = nn.Linear(cond_depth, depth_out * 4)

    def _init_directional_convs(self):
        raise NotImplementedError

    def _run_stacks(self, vert_in, horiz_in, conds):
        vert_out = self._run_padded_vertical(vert_in)
        horiz_out = self._run_padded_horizontal(horiz_in)
        horiz_out = horiz_out + self.vert_to_horiz(vert_out)

        if conds is not None:
            cond_bias = self._compute_cond_bias(conds)
            vert_out = vert_out + cond_bias[:, :self.depth_out*2]
            horiz_out = horiz_out + cond_bias[:, self.depth_out*2:]

        vert_out = gated_activation(vert_out)
        horiz_out = gated_activation(horiz_out)
        return vert_out, horiz_out

    def _run_padded_vertical(self, vert_in):
        raise NotImplementedError

    def _run_padded_horizontal(self, horiz_in):
        raise NotImplementedError

    def _compute_cond_bias(self, conds):
        if len(conds.shape) == 2:
            outputs = self.cond_layer(conds)
            return outputs.view(-1, outputs.shape[1], 1, 1)
        assert len(conds.shape) == 4
        conds_perm = conds.permute(0, 2, 3, 1)
        outputs = self.cond_layer(conds_perm)
        return outputs.permute(0, 3, 1, 2)


class PixelConvA(PixelConv):
    """
    The first layer in a PixelCNN. This layer is unlike
    the other layers, in that it does not allow the stack
    to see the current pixel.

    Args:
        depth_in: the number of input filters.
        depth_out: the number of output filters.
        cond_depth: the number of conditioning channels.
          If None, this is an unconditional model.
        horizontal: the receptive field of the horizontal
          stack.
        vertical: the receptive field of the vertical
          stack.
    """

    def __init__(self, depth_in, depth_out, cond_depth=None, horizontal=2, vertical=2):
        super().__init__(depth_in, depth_out, cond_depth=cond_depth, horizontal=2, vertical=2)

    def forward(self, images, conds=None):
        """
        Apply the layer to some images, producing latents.

        Args:
            images: an NCHW batch of images.
            conds: an optional conditioning value. If set,
              either an NCHW Tensor or an NxM Tensor.

        Returns:
            A tuple (vertical, horizontal), one for each
              of the two directional stacks.
        """
        return self._run_stacks(images, images, conds)

    def _init_directional_convs(self):
        self.vertical_conv = nn.Conv2d(self.depth_in, self.depth_out * 2,
                                       (self.vertical, self.horizontal*2 + 1))
        self.horizontal_conv = nn.Conv2d(self.depth_in, self.depth_out * 2, (1, self.horizontal))

    def _run_padded_vertical(self, vert_in):
        vert_pad = (self.horizontal, self.horizontal, self.vertical, 0)
        return self.vertical_conv(F.pad(vert_in, vert_pad))[:, :, :-1, :]

    def _run_padded_horizontal(self, horiz_in):
        return self.horizontal_conv(F.pad(horiz_in, (self.horizontal, 0, 0, 0)))[:, :, :, :-1]


class PixelConvB(PixelConv):
    """
    Any layer except the first in a PixelCNN.

    Args:
        depth_in: the number of input filters.
        cond_depth: the number of conditioning channels.
          If None, this is an unconditional model.
        horizontal: the receptive field of the horizontal
          stack.
        vertical: the receptive field of the vertical
          stack.
    """

    def __init__(self, depth_in, cond_depth=None, norm=False, horizontal=2, vertical=2):
        super().__init__(depth_in, depth_in, cond_depth=cond_depth, horizontal=horizontal,
                         vertical=vertical)
        self.horiz_residual = nn.Conv2d(depth_in, depth_in, 1)
        self.vert_norm = lambda x: x
        self.horiz_norm = lambda x: x
        if norm:
            self.vert_norm = ChannelNorm(depth_in)
            self.horiz_norm = ChannelNorm(depth_in)

    def forward(self, vert_in, horiz_in, conds=None):
        """
        Apply the layer to the outputs of previous
        vertical and horizontal stacks.

        Args:
            vert_in: an NCHW Tensor.
            horiz_in: an NCHW Tensor.
            conds: an optional conditioning value. If set,
              either an NCHW Tensor or an NxM Tensor.

        Returns:
            A tuple (vertical, horizontal), one for each
              of the two directional stacks.
        """
        vert_out, horiz_out = self._run_stacks(vert_in, horiz_in, conds)
        horiz_out = horiz_in + self.horiz_norm(self.horiz_residual(horiz_out))
        return self.vert_norm(vert_out), horiz_out

    def _init_directional_convs(self):
        self.vertical_conv = nn.Conv2d(self.depth_in, self.depth_out * 2,
                                       (self.vertical + 1, self.horizontal*2 + 1))
        self.horizontal_conv = nn.Conv2d(self.depth_in, self.depth_out * 2,
                                         (1, self.horizontal + 1))

    def _run_padded_vertical(self, vert_in):
        vert_pad = (self.horizontal, self.horizontal, self.vertical, 0)
        return self.vertical_conv(F.pad(vert_in, vert_pad))

    def _run_padded_horizontal(self, horiz_in):
        return self.horizontal_conv(F.pad(horiz_in, (self.horizontal, 0, 0, 0)))


class ChannelNorm(nn.Module):
    """
    A layer which applies layer normalization to the
    channels at each spacial location separately.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm((num_channels,))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def gated_activation(outputs):
    depth = outputs.shape[1] // 2
    tanh = torch.tanh(outputs[:, :depth])
    sigmoid = torch.sigmoid(outputs[:, depth:])
    return tanh * sigmoid
