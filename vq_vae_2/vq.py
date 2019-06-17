"""
Vector-Quantization for the VQ-VAE itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ(nn.Module):
    """
    A vector quantization layer.

    This layer takes continuous inputs and produces a few
    different types of outputs, including a discretized
    output, a commitment loss, a codebook loss, etc.

    Args:
        num_channels: the depth of the input Tensors.
        num_latents: the number of latent values in the
          dictionary to choose from.
        dead_rate: the number of forward passes after
          which a dictionary entry is considered dead if
          it has not been used.
    """

    def __init__(self, num_channels, num_latents, dead_rate=100):
        self.num_channels = num_channels
        self.num_latents = num_latents
        self.dead_rate = dead_rate

        self.dictionary = nn.Parameter(torch.randn(num_latents, num_channels))
        self.usage_count = nn.Parameter(dead_rate * torch.ones(num_latents).long(),
                                        requires_grad=False)

    def forward(self, inputs):
        """
        Apply vector quantization.

        If the module is in training mode, this will also
        update the usage tracker and re-initialize dead
        dictionary entries.

        Args:
            inputs: the input Tensor. Either [N x C] or
              [N x C x H x W].

        Returns:
            A tuple (embedded, embedded_pt, idxs):
              embedded: the new [N x C x H x W] Tensor
                which passes gradients to the dictionary.
              embedded_pt: like embedded, but with a
                passthrough gradient estimator. Gradients
                through this pass directly to the inputs.
              idxs: a [N x H x W] Tensor of Longs
                indicating the chosen dictionary entries.
        """
        channels_last = inputs
        if len(inputs.shape) == 4:
            # NCHW to NHWC
            channels_last = inputs.permute(0, 2, 3, 1).contiguous()

        diffs = embedding_distances(channels_last)
        idxs = torch.argmin(diffs, dim=-1)
        embedded = F.embedding(idxs, self.dictionary)

        if len(inputs.shape) == 4:
            # NHWC to NCHW
            embedded = embedded.permute(0, 3, 1, 2).contiguous()

        embedded_pt = embedded.detach() + (inputs - inputs.detach())

        if self.training:
            self._update_tracker(idxs)
            self._reinit_dead_centers(inputs)

        return embedded, embedded_pt, idxs

    def _update_tracker(self, idxs):
        # TODO: this.
        pass

    def _reinit_dead_centers(self, inputs):
        # TODO: this.
        pass


def embedding_distances(dictionary, tensor):
    """
    Compute distances between every embedding in a
    dictionary and every vector in a Tensor.

    This will not generate a huge intermediate Tensor,
    unlike the naive implementation.

    Args:
        dictionary: a [D x C] Tensor.
        tensor: a [... x C] Tensor.

    Returns:
        A [... x D] Tensor of distances.
    """
    dict_norms = torch.sum(torch.pow(dictionary, 2), dim=-1)
    tensor_norms = torch.sum(torch.pow(tensor, 2), dim=-1)
    dots = torch.matmul(dictionary, tensor[..., None])[..., 0]
    return -2 * dots + dict_norms + tensor_norms[..., None]
