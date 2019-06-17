"""
Train a PixelCNN on MNIST using a pre-trained VQ-VAE.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms

from vq_vae_2.examples.mnist.model import Encoder, Generator

BATCH_SIZE = 32
LR = 1e-3


def main():
    encoder = Encoder()
    encoder.load_state_dict(torch.load('enc.pt'))
    encoder.eval()

    generator = Generator()
    if os.path.exists('gen.pt'):
        generator.load_state_dict(torch.load('gen.pt'))

    optimizer = optim.Adam(generator.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for batch_idx, images in enumerate(load_images()):
        _, (_, _, encoded) = encoder(images)
        logits = generator(encoded)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_fn(logits, encoded.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss=%f' % loss.item())
        if not batch_idx % 100:
            torch.save(generator.state_dict(), 'gen.pt')


def load_images():
    while True:
        for data, _ in create_data_loader():
            yield data


def create_data_loader():
    mnist = torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    main()
