"""
Train a hierarchical VAE on text data.
"""

import argparse
import os

import torch
import torch.optim as optim

from vq_vae_2.examples.text.data import load_text_samples
from vq_vae_2.examples.text.model import make_vae

VAE_PATH = 'vae.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    if os.path.exists(VAE_PATH):
        vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    for i, batch in enumerate(load_text_samples(args.data, args.batch_size, args.context_len)):
        batch = batch.to(device)
        terms = vae(batch)
        print('step %d: loss=%f' % (i, terms['losses'][-1].item()))
        optimizer.zero_grad()
        terms['loss'].backward()
        optimizer.step()
        vae.revive_dead_entries()
        if not i % 100:
            torch.save(vae.state_dict(), VAE_PATH)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='torch device', default='cuda')
    parser.add_argument('--batch-size', help='batch size', default=16, type=int)
    parser.add_argument('--context-len', help='context size in bytes', default=512, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--save-interval', help='steps per model save', default=30, type=int)
    parser.add_argument('data', help='data file')
    return parser


if __name__ == '__main__':
    main()
