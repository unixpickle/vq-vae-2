"""
Models for hierarchical image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_vae_2.attention import PixelAttention
from vq_vae_2.pixel_cnn import PixelCNN, PixelConvA, PixelConvB
from vq_vae_2.vq_vae import HalfDecoder, HalfQuarterDecoder, HalfEncoder, QuarterEncoder, VQVAE


def make_vae():
    encoders = [QuarterEncoder(3, 128, 512), HalfEncoder(128, 128, 512)]
    decoders = [HalfDecoder(128, 128), HalfQuarterDecoder(128, 3)]
    return VQVAE(encoders, decoders)


class TopPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(512, 512)
        self.pixel_cnn = PixelCNN(
            PixelConvA(512, 512),

            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelAttention(512),

            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelAttention(512),

            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelAttention(512),

            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelConvB(512),
            PixelAttention(512),
        )
        self.out_stack = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            Residual1x1(512),
            Residual1x1(512),
            Residual1x1(512),
            Residual1x1(512),
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.pixel_cnn(x)
        return self.out_stack(torch.cat([out1, out2], dim=1))


class Residual1x1(nn.Module):
    def __init__(self, num_channels):
        self.conv1 = nn.Conv2d(num_channels, num_channels, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return inputs + x
