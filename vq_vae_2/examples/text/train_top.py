"""
Train the top-level prior.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from vq_vae_2.examples.text.data import load_text_samples
from vq_vae_2.examples.text.model import TopPrior, make_vae
from vq_vae_2.examples.text.train_vae import VAE_PATH, arg_parser

TOP_PRIOR_PATH = 'top.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)
    vae.eval()

    top_prior = TopPrior(args.context_len // 8)
    if os.path.exists(TOP_PRIOR_PATH):
        top_prior.load_state_dict(torch.load(TOP_PRIOR_PATH, map_location='cpu'))
    top_prior.to(device)

    optimizer = optim.Adam(top_prior.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for i, batch in enumerate(load_text_samples(args.data, args.batch_size, args.context_len)):
        batch = batch.to(device)
        enc_1 = vae.encoders[0].encode(batch)
        enc_2 = vae.encoders[1].encode(enc_1)
        _, _, quantized = vae.encoders[2](enc_2)
        logits = top_prior(quantized)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_fn(logits, quantized.view(-1))
        print('step %d: loss=%f' % (i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % args.save_interval:
            torch.save(top_prior.state_dict(), TOP_PRIOR_PATH)


if __name__ == '__main__':
    main()
