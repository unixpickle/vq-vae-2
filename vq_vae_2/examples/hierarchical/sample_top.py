"""
Generate samples using the top-level prior.
"""

import argparse
import random

from PIL import Image
import numpy as np
import torch

from vq_vae_2.examples.hierarchical.model import TopPrior, make_vae
from vq_vae_2.examples.hierarchical.train_top import TOP_PRIOR_PATH
from vq_vae_2.examples.hierarchical.train_vae import VAE_PATH

NUM_SAMPLES = 4


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    vae.load_state_dict(torch.load(VAE_PATH))
    vae.to(device)
    vae.eval()

    top_prior = TopPrior()
    top_prior.load_state_dict(torch.load(TOP_PRIOR_PATH))
    top_prior.to(device)

    results = np.zeros([NUM_SAMPLES, 32, 32], dtype=np.long)
    for row in range(results.shape[1]):
        for col in range(results.shape[2]):
            partial_in = torch.from_numpy(results[:, :row + 1]).to(device)
            with torch.no_grad():
                outputs = torch.softmax(top_prior(partial_in), dim=1).cpu().numpy()
            for i, out in enumerate(outputs):
                probs = out[:, row, col]
                results[i, row, col] = sample_softmax(probs)
        print('done row', row)
    with torch.no_grad():
        full_latents = torch.from_numpy(results).to(device)
        top_embedded = vae.encoders[1].vq.embed(full_latents)
        bottom_encoded = vae.decoders[0]([top_embedded])
        bottom_embedded, _, _ = vae.encoders[0].vq(bottom_encoded)
        decoded = torch.clamp(vae.decoders[1]([top_embedded, bottom_embedded]), 0, 1)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    decoded = np.concatenate(decoded, axis=1)
    Image.fromarray((decoded * 255).astype(np.uint8)).save('top_samples.png')


def sample_softmax(probs):
    number = random.random()
    for i, x in enumerate(probs):
        number -= x
        if number <= 0:
            return i
    return len(probs) - 1


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='torch device', default='cuda')
    return parser


if __name__ == '__main__':
    main()
