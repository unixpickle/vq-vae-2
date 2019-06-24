"""
Train the bottom-level prior.
"""

import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim

from vq_vae_2.examples.hierarchical.data import load_images
from vq_vae_2.examples.hierarchical.model import BottomPrior, make_vae
from vq_vae_2.examples.hierarchical.train_vae import VAE_PATH, arg_parser

BOTTOM_PRIOR_PATH = 'bottom.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)
    vae.eval()

    bottom_prior = BottomPrior()
    if os.path.exists(BOTTOM_PRIOR_PATH):
        bottom_prior.load_state_dict(torch.load(BOTTOM_PRIOR_PATH, map_location='cpu'))
    bottom_prior.to(device)

    optimizer = optim.Adam(bottom_prior.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    data = load_images(args.data, batch_size=2)
    for i in itertools.count():
        images = next(data).to(device)
        bottom_enc = vae.encoders[0].encode(images)
        _, _, bottom_idxs = vae.encoders[0].vq(bottom_enc)
        _, _, top_idxs = vae.encoders[1](bottom_enc)
        logits = bottom_prior(bottom_idxs, top_idxs)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_fn(logits, bottom_idxs.view(-1))
        print('step %d: loss=%f' % (i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 30:
            torch.save(bottom_prior.state_dict(), BOTTOM_PRIOR_PATH)


if __name__ == '__main__':
    main()
