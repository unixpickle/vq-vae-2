"""
Train a PixelCNN on MNIST.
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms

from vq_vae_2.examples.mnist.model import Model

BATCH_SIZE = 32
LR = 1e-3


def main():
    loader = create_data_loader()
    model = Model()
    if os.path.exists('pixel_cnn.pt'):
        model.load_state_dict(torch.load('pixel_cnn.pt'))
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for batch_idx, (data, _) in enumerate(loader):
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss=%f' % loss.item())
        torch.save(model.state_dict(), 'pixel_cnn.pt')


def create_data_loader():
    mnist = torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    main()
