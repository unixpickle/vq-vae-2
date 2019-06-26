"""
An implementation of the hierarchical VQ-VAE.
See https://arxiv.org/abs/1906.00446.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VQ, vq_loss


class Encoder(nn.Module):
    """
    An abstract VQ-VAE encoder, which takes input Tensors,
    shrinks them, and quantizes the result.

    Sub-classes should overload the encode() method.

    Args:
        num_channels: the number of channels in the latent
          codebook.
        num_latents: the number of entries in the latent
          codebook.
        kwargs: arguments to pass to the VQ layer.
    """

    def __init__(self, num_channels, num_latents, **kwargs):
        super().__init__()
        self.vq = VQ(num_channels, num_latents, **kwargs)

    def encode(self, x):
        """
        Encode a Tensor before the VQ layer.

        Args:
            x: the input Tensor.

        Returns:
            A Tensor with the correct number of output
              channels (according to self.vq).
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Apply the encoder.

        See VQ.forward() for return values.
        """
        return self.vq(self.encode(x))


class QuarterEncoder(Encoder):
    """
    The encoder from the original VQ-VAE paper that cuts
    the dimensions down by a factor of 4 in both
    directions.
    """

    def __init__(self, in_channels, out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 4, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 4, stride=2)
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def encode(self, x):
        # Padding is uneven, so we make the right and
        # bottom more padded arbitrarily.
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv1(x)
        x = F.relu(x)
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv2(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x


class HalfEncoder(Encoder):
    """
    An encoder that cuts the input size in half in both
    dimensions.
    """

    def __init__(self, in_channels, out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def encode(self, x):
        x = self.conv(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x


class Decoder(nn.Module):
    """
    An abstract VQ-VAE decoder, which takes a stack of
    (differently-sized) input Tensors and produces a
    predicted output Tensor.

    Sub-classes should overload the forward() method.
    """

    def forward(self, inputs):
        """
        Apply the decoder to a list of inputs.

        Args:
            inputs: a sequence of input Tensors. There may
              be more than one in the case of a hierarchy,
              in which case the top levels come first.

        Returns:
            A decoded Tensor.
        """
        raise NotImplementedError


class QuarterDecoder(Decoder):
    """
    The decoder from the original VQ-VAE paper that
    upsamples the dimensions by a factor of 4 in both
    directions.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class HalfDecoder(Decoder):
    """
    A decoder that upsamples by a factor of 2 in both
    dimensions.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv(x)
        return x


class HalfQuarterDecoder(Decoder):
    """
    A decoder that takes two inputs. The first one is
    upsampled by a factor of two, and then combined with
    the second input which is further upsampled by a
    factor of four.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3 = _make_residual(in_channels)
        self.residual4 = _make_residual(in_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 2

        # Upsample the top input to match the shape of the
        # bottom input.
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)

        # Mix together the bottom and top inputs.
        x = torch.cat([x, inputs[1]], dim=1)
        x = self.conv2(x)

        x = x + self.residual3(x)
        x = x + self.residual4(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x


class VQVAE(nn.Module):
    """
    A complete VQ-VAE hierarchy.

    There are N encoders, stored from the bottom level to
    the top level, and N decoders stored from top to
    bottom.
    """

    def __init__(self, encoders, decoders):
        super().__init__()
        assert len(encoders) == len(decoders)
        self.encoders = encoders
        self.decoders = decoders
        for i, enc in enumerate(encoders):
            self.add_module('encoder_%d' % i, enc)
        for i, dec in enumerate(decoders):
            self.add_module('decoder_%d' % i, dec)

    def forward(self, inputs, commitment=0.25):
        """
        Compute training losses for a batch of inputs.

        Args:
            inputs: the input Tensor. If this is a Tensor
              of integers, then cross-entropy loss will be
              used for the final decoder. Otherwise, MSE
              will be used.
            commitment: the commitment loss coefficient.

        Returns:
            A dict of Tensors, containing at least:
              loss: the total training loss.
              losses: the MSE/log-loss from each decoder.
              reconstructions: a reconstruction Tensor
                from each decoder.
              embedded: outputs from every encoder, passed
                through the vector-quantization table.
                Ordered from bottom to top level.
        """
        all_encoded = [inputs]
        all_vq_outs = []
        total_vq_loss = 0.0
        total_recon_loss = 0.0
        for encoder in self.encoders:
            encoded = encoder.encode(all_encoded[-1])
            embedded, embedded_pt, _ = encoder.vq(encoded)
            all_encoded.append(encoded)
            all_vq_outs.append(embedded_pt)
            total_vq_loss = total_vq_loss + vq_loss(encoded, embedded, commitment=commitment)
        losses = []
        reconstructions = []
        for i, decoder in enumerate(self.decoders):
            dec_inputs = all_vq_outs[::-1][:i + 1]
            target = all_encoded[::-1][i + 1]
            recon = decoder(dec_inputs)
            reconstructions.append(recon)
            if target.dtype.is_floating_point:
                recon_loss = torch.mean(torch.pow(recon - target.detach(), 2))
            else:
                recon_loss = F.cross_entropy(recon.view(-1, recon.shape[-1]), target.view(-1))
            total_recon_loss = total_recon_loss + recon_loss
            losses.append(recon_loss)
        return {
            'loss': total_vq_loss + total_recon_loss,
            'losses': losses,
            'reconstructions': reconstructions,
            'embedded': all_vq_outs,
        }

    def revive_dead_entries(self):
        """
        Revive dead entries from all of the VQ layers.

        Only call this once the encoders have all been
        through a forward pass in training mode.
        """
        for enc in self.encoders:
            enc.vq.revive_dead_entries()

    def full_reconstructions(self, inputs):
        """
        Compute reconstructions of the inputs using all
        the different layers of the hierarchy.

        The first reconstruction uses only information
        from the top-level codes, the second uses only
        information from the top-level and second-to-top
        level codes, etc.

        This is not forward(inputs)['reconstructions'],
        since said reconstructions are simply each level's
        reconstruction of the next level's features.
        Instead, full_reconstructions reconstructs the
        original inputs.
        """
        terms = self(inputs)
        layer_recons = []
        for encoder, recon in zip(self.encoders[:-1][::-1], terms['reconstructions'][:-1]):
            _, embedded_pt, _ = encoder.vq(recon)
            layer_recons.append(embedded_pt)
        hierarchy_size = len(self.decoders)
        results = []
        for i in range(hierarchy_size - 1):
            num_actual = i + 1
            dec_in = terms['embedded'][-num_actual:][::-1] + layer_recons[num_actual - 1:]
            results.append(self.decoders[-1](dec_in))
        results.append(terms['reconstructions'][-1])
        return results


def _make_residual(channels):
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 1),
    )
