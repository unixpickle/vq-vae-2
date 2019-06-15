"""
A basic PixelCNN model.
"""

import torch.nn as nn

from vq_vae_2.pixel_cnn import PixelCNN, PixelConvA, PixelConvB


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PixelCNN(
            PixelConvA(1, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
            PixelConvB(64, 64),
        )
        self.to_logits = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        out1, out2 = self.model(x)
        return self.to_logits(out1 + out2)
