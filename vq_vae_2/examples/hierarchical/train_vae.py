"""
Train a hierarchical VQ-VAE on 256x256 images.
"""

import argparse
import itertools
import os

from PIL import Image
import numpy as np
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
        images = next(data).to(device)
        terms = model(images)
        print('step %d: mse=%f mse_top=%f' %
              (i, terms['mse'][-1].item(), terms['mse'][0].item()))
        optimizer.zero_grad()
        terms['loss'].backward()
        optimizer.step()
        if not i % 30:
            torch.save(model.state_dict(), VAE_PATH)
            save_reconstructions(model, images, terms)


def save_reconstructions(vae, images, terms):
    real_recons = torch.clamp(terms['reconstructions'][-1], 0, 1)
    real_recons = real_recons.permute(0, 2, 3, 1).detach().cpu().numpy()

    # Create reconstructions using only the top latents.
    top_embed, _, _ = vae.encoders[1](vae.encoders[0].encode(images))
    bottom_embed, _, _ = vae.encoders[0].vq(terms['reconstructions'][0])
    top_recons = torch.clamp(vae.decoders[1]([top_embed, bottom_embed]), 0, 1)
    top_recons = top_recons.permute(0, 2, 3, 1).detach().cpu().numpy()

    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    columns = np.concatenate([top_recons, real_recons, images], axis=-2)
    columns = np.concatenate(columns, axis=0)
    Image.fromarray((columns * 255).astype('uint8')).save('reconstructions.png')


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data directory')
    parser.add_argument('--device', help='torch device', default='cuda')
    return parser


if __name__ == '__main__':
    main()
