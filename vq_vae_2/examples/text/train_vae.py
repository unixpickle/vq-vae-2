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
MAX_LOSS_GAIN = 3


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    if os.path.exists(VAE_PATH):
        vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=1e-4)

    last_loss = None
    last_state = None
    for i, batch in enumerate(load_text_samples(args.data, args.batch_size, args.context_len)):
        state = vae.state_dict()
        batch = batch.to(device)
        terms = vae(batch)
        loss = terms['loss'][-1].item()
        if last_loss is not None and loss > MAX_LOSS_GAIN * last_loss:
            vae.load_state_dict(last_state)
            print('step %d: reset with loss %f' % (i, loss))
            continue
        last_loss = loss
        last_state = state
        print('step %d: loss=%f' % (i, loss))
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
    parser.add_argument('data', help='data file')
    return parser


if __name__ == '__main__':
    main()
