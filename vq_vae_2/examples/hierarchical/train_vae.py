"""
Train a hierarchical VQ-VAE on 256x256 images.
"""

import argparse
import itertools
import os

import torch
import torch.optim as optim

from vq_vae_2.examples.hierarchical.data import load_images
from vq_vae_2.examples.hierarchical.model import make_vae

VAE_PATH = 'vae.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)
    model = make_vae()
    if os.path.exists(VAE_PATH):
        model.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    data = load_images(args.data)
    for i in itertools.count():
        terms = model(next(data).to(device))
        print('step %d: mse=%f mse_top=%f' %
              (i, terms['mse'][-1].item(), terms['mse'][0].item()))
        optimizer.zero_grad()
        terms['loss'].backward()
        optimizer.step()
        if not i % 10:
            torch.save(model.state_dict(), VAE_PATH)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data directory')
    parser.add_argument('--device', help='torch device', default='cuda')
    return parser


if __name__ == '__main__':
    main()
