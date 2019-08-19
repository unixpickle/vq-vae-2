"""
Models for compressing natural language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_vae_2.vq_vae import Decoder, Encoder, VQVAE
from vq_vae_2.attention import MaskedAttention

DEAD_RATE = 100


def make_vae():
    return VQVAE([BottomEncoder(), HighEncoder(), HighEncoder()],
                 [HierarchyDecoder(512, 512, 1), HierarchyDecoder(512, 512, 2),
                  HierarchyDecoder(512, 256, 3)])


class BottomEncoder(Encoder):
    def __init__(self, num_channels=512, num_latents=512):
        super().__init__(num_channels, num_latents, dead_rate=DEAD_RATE)
        self.embed = nn.Embedding(256, num_channels)
        self.stack = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_channels, num_channels, 3, stride=2, padding=1),
            Residual1d(num_channels),
            Residual1d(num_channels),
        )

    def encode(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.stack(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class HighEncoder(Encoder):
    def __init__(self, num_channels=512, num_latents=512):
        super().__init__(num_channels, num_latents, dead_rate=DEAD_RATE)
        self.stack = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_channels, num_channels, 3, stride=2, padding=1),
            Residual1d(num_channels),
            Residual1d(num_channels),
        )

    def encode(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.stack(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class HierarchyDecoder(Decoder):
    def __init__(self, in_channels, out_channels, num_inputs=1):
        super().__init__()
        self.layers = []
        for i in range(num_inputs):
            stack = nn.Sequential(
                Residual1d(in_channels),
                Residual1d(in_channels),
                nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1),
            )
            self.add_module('hierarchy%d' % i, stack)
            self.layers.append(stack)
        self.conv_out = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, inputs):
        full_inputs = None
        for stack, x in zip(self.layers, inputs):
            if full_inputs is not None:
                x = x + full_inputs
            x = x.permute(0, 2, 1).contiguous()
            x = stack(x)
            x = x.permute(0, 2, 1).contiguous()
            full_inputs = x
        out = full_inputs.permute(0, 2, 1).contiguous()
        out = self.conv_out(out)
        out = out.permute(0, 2, 1).contiguous()
        return out


class TopPrior(nn.Module):
    def __init__(self, seq_len, depth=256, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(512, depth)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, depth))
        self.attention = nn.Sequential(
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
        )
        self.out_stack = nn.Sequential(
            nn.Conv1d(depth, depth, 1),
            nn.ReLU(),
            nn.Conv1d(depth, depth, 1),
            nn.ReLU(),
            nn.Conv1d(depth, 512, 1),
        )

    def forward(self, x):
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x = self.embed(x)
        x = x + self.pos_enc
        x = self.attention(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.out_stack(x)
        return x


class LowPrior(nn.Module):
    def __init__(self, num_inputs, seq_len, depth=512, num_heads=8):
        super().__init__()
        self.embeddings = []
        self.layers = []
        for i in range(num_inputs - 1):
            embed = nn.Embedding(512, depth)
            stack = nn.Sequential(
                Residual1d(depth),
                Residual1d(depth),
                Residual1d(depth),
                nn.ConvTranspose1d(depth, depth, 4, stride=2, padding=1),
                Residual1d(depth),
            )
            self.add_module('embed%d' % i, embed)
            self.add_module('stack%d' % i, stack)
            self.embeddings.append(embed)
            self.layers.append(stack)
        self.embed = nn.Embedding(512, depth)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, depth))
        self.attention = nn.Sequential(
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
            AttentionLayer(depth, num_heads),
        )
        self.out_stack = nn.Sequential(
            nn.Conv1d(depth, depth, 1),
            nn.ReLU(),
            nn.Conv1d(depth, depth, 1),
            nn.ReLU(),
            nn.Conv1d(depth, 512, 1),
        )

    def forward(self, *inputs):
        cond = None
        for embed, layer, x in zip(self.embeddings, self.layers, inputs):
            em = embed(x).permute(0, 2, 1).contiguous()
            if cond is None:
                cond = layer(em)
            else:
                cond = layer(cond + em)
        cond = cond.permute(0, 2, 1).contiguous()
        x = inputs[-1]
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x = self.embed(x)
        x = x + cond
        x = x + self.pos_enc
        x = self.attention(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.out_stack(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, depth, num_heads, hidden=2048):
        super().__init__()
        self.attention = MaskedAttention(depth, num_heads=num_heads)
        self.fc1 = nn.Conv1d(depth, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, depth, 1)
        self.norm1 = nn.LayerNorm(depth)
        self.norm2 = nn.LayerNorm(depth)

    def forward(self, x):
        original = x
        x = self.attention(x)
        x = self.norm1(x + original)

        original = x
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1).contiguous()
        return self.norm2(x + original)


class Residual1d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Tanh(),
            nn.Conv1d(num_channels, num_channels, 3, padding=1),
            nn.Tanh(),
            nn.Conv1d(num_channels, num_channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.stack(x)
