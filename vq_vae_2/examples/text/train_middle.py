"""
Train the top-level prior.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from vq_vae_2.examples.text.data import load_text_samples
from vq_vae_2.examples.text.model import LowPrior, make_vae
from vq_vae_2.examples.text.train_vae import VAE_PATH, arg_parser

MIDDLE_PRIOR_PATH = 'middle.pt'


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)
    vae.eval()

    prior = LowPrior(2)
    if os.path.exists(MIDDLE_PRIOR_PATH):
        prior.load_state_dict(torch.load(MIDDLE_PRIOR_PATH, map_location='cpu'))
    prior.to(device)

    optimizer = optim.Adam(prior.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for i, batch in enumerate(load_text_samples(args.data, args.batch_size, args.context_len)):
        batch = batch.to(device)
        enc_1 = vae.encoders[0].encode(batch)
        enc_2, _, enc_2_quantized = vae.encoders[1](enc_1)
        _, _, enc_3_quantized = vae.encoders[2](enc_2)
        logits = prior(enc_3_quantized, enc_2_quantized)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_fn(logits, enc_2_quantized.view(-1))
        print('step %d: loss=%f' % (i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 30:
            torch.save(prior.state_dict(), MIDDLE_PRIOR_PATH)


if __name__ == '__main__':
    main()
