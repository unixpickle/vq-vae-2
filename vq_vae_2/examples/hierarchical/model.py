"""
Models for hierarchical image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_vae_2.attention import PixelAttention
from vq_vae_2.pixel_cnn import ChannelNorm, PixelCNN, PixelConvA, PixelConvB
from vq_vae_2.vq_vae import HalfDecoder, HalfQuarterDecoder, HalfEncoder, QuarterEncoder, VQVAE


def make_vae():
    encoders = [QuarterEncoder(3, 128, 512), HalfEncoder(128, 128, 512)]
    decoders = [HalfDecoder(128, 128), HalfQuarterDecoder(128, 3)]
    return VQVAE(encoders, decoders)


class TopPrior(nn.Module):
    def __init__(self, depth=128, num_heads=2):
        super().__init__()
        self.embed = nn.Embedding(512, depth)
        self.pixel_cnn = PixelCNN(
            PixelConvA(depth, depth),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),

            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelConvB(depth, norm=True),
            PixelAttention(depth, num_heads=num_heads),
        )
        self.out_stack = nn.Sequential(
            nn.Conv2d(depth * 2, depth, 1),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            Residual1x1(depth),
            nn.Conv2d(depth, 512, 1),
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.pixel_cnn(x)
        return self.out_stack(torch.cat([out1, out2], dim=1))


class BottomPrior(nn.Module):
    def __init__(self, depth=128, num_heads=2):
        super().__init__()
        self.embed_top = nn.Embedding(512, depth)
        self.embed_bottom = nn.Embedding(512, depth)
        self.cond_stack = nn.Sequential(
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            Residual3x3(depth),
            nn.ConvTranspose2d(depth, depth, 4, stride=2, padding=1),
        )
        self.pixel_cnn = PixelCNN(
            PixelConvA(depth, depth, cond_depth=depth),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),

            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
            PixelConvB(depth, cond_depth=depth, norm=True),
        )
        self.out_stack = nn.Sequential(
            nn.Conv2d(depth * 2, depth, 1),
            nn.Conv2d(depth, 512, 1),
        )

    def forward(self, bottom, top):
        conds = self.embed_top(top)
        conds = conds.permute(0, 3, 1, 2).contiguous()
        conds = self.cond_stack(conds)

        out = self.embed_bottom(bottom)
        out = out.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.pixel_cnn(out, conds=conds)
        return self.out_stack(torch.cat([out1, out2], dim=1))


class Residual1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 1)
        self.norm = ChannelNorm(num_channels)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return inputs + self.norm(x)


class Residual3x3(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.norm = ChannelNorm(num_channels)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return inputs + self.norm(x)
