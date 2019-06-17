"""
A basic PixelCNN model.
"""

import torch.nn as nn

from vq_vae_2.pixel_cnn import PixelCNN, PixelConvA, PixelConvB
from vq_vae_2.vq import VQ

LATENT_SIZE = 16
LATENT_COUNT = 32


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, LATENT_SIZE, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(LATENT_SIZE, LATENT_SIZE, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(LATENT_SIZE, LATENT_SIZE, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(LATENT_SIZE, LATENT_SIZE, 3, padding=1),
        )
        self.vq = VQ(LATENT_SIZE, LATENT_COUNT)

    def forward(self, x):
        x = self.layers(x)
        return x, self.vq(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(LATENT_SIZE, LATENT_SIZE, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(LATENT_SIZE, LATENT_SIZE, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(LATENT_SIZE, LATENT_SIZE, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(LATENT_SIZE, LATENT_SIZE, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(LATENT_SIZE, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(LATENT_COUNT, 64)
        self.model = PixelCNN(
            PixelConvA(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
        )
        self.to_logits = nn.Conv2d(64, LATENT_COUNT, 1)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        out1, out2 = self.model(x)
        return self.to_logits(out1 + out2)
