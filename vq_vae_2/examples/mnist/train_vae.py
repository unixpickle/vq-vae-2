"""
Train an encoder/decoder on the MNIST dataset.
"""

import os

from PIL import Image
import numpy as np
import torch
import torch.optim as optim

from vq_vae_2.examples.mnist.model import make_vq_vae
from vq_vae_2.examples.mnist.train_generator import load_images


DEVICE = torch.device('cpu')


def main():
    vae = make_vq_vae()
    if os.path.exists('vae.pt'):
        vae.load_state_dict(torch.load('vae.pt', map_location='cpu'))
    vae.to(DEVICE)
    optimizer = optim.Adam(vae.parameters())
    for i, batch in enumerate(load_images()):
        batch = batch.to(DEVICE)
        terms = vae(batch)
        print('step %d: loss=%f mse=%f' %
              (i, terms['loss'].item(), terms['mse'][-1].item()))
        optimizer.zero_grad()
        terms['loss'].backward()
        optimizer.step()
        vae.revive_dead_entries()
        if not i % 10:
            torch.save(vae.state_dict(), 'vae.pt')
        if not i % 100:
            save_reconstructions(batch, terms['reconstructions'][-1])


def save_reconstructions(batch, decoded):
    batch = batch.detach().permute(0, 2, 3, 1).contiguous()
    decoded = decoded.detach().permute(0, 2, 3, 1).contiguous()
    input_images = (np.concatenate(batch.cpu().numpy(), axis=0) * 255).astype(np.uint8)
    output_images = np.concatenate(decoded.cpu().numpy(), axis=0)
    output_images = (np.clip(output_images, 0, 1) * 255).astype(np.uint8)
    joined = np.concatenate([input_images[..., 0], output_images[..., 0]], axis=1)
    Image.fromarray(joined).save('reconstructions.png')


if __name__ == '__main__':
    main()
