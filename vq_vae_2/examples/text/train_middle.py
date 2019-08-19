"""
Train the mid-level prior.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from vq_vae_2.examples.text.data import load_text_samples
from vq_vae_2.examples.text.model import LowPrior, make_vae
from vq_vae_2.examples.text.train_vae import VAE_PATH, arg_parser

MIDDLE_PRIOR_PATH = 'middle.pt'


def main(prior_path=MIDDLE_PRIOR_PATH, num_levels=2):
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)
    vae.eval()

    prior = LowPrior(num_levels, args.context_len >> (4 - num_levels))
    if os.path.exists(prior_path):
        prior.load_state_dict(torch.load(prior_path, map_location='cpu'))
    prior.to(device)

    optimizer = optim.Adam(prior.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for i, batch in enumerate(load_text_samples(args.data, args.batch_size, args.context_len)):
        inputs = batch.to(device)
        quantized = []
        for encoder in vae.encoders:
            inputs, _, quant = encoder(inputs)
            quantized.append(quant)
        logits = prior(*quantized[::-1][:num_levels])
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_fn(logits, quantized[-num_levels].view(-1))
        print('step %d: loss=%f' % (i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % args.save_interval:
            torch.save(prior.state_dict(), prior_path)


if __name__ == '__main__':
    main()
