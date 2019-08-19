"""
Produce samples from all the priors combined.
"""

import os

import numpy as np
import torch

from vq_vae_2.examples.text.data import load_text_samples
from vq_vae_2.examples.text.model import LowPrior, make_vae
from vq_vae_2.examples.text.recon_vae import print_bytes
from vq_vae_2.examples.text.sample_top import arg_parser, sample_softmax
from vq_vae_2.examples.text.train_bottom import BOTTOM_PRIOR_PATH
from vq_vae_2.examples.text.train_vae import VAE_PATH


def main():
    parser = arg_parser()
    parser.add_argument('data', help='data file')
    args = parser.parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    if os.path.exists(VAE_PATH):
        vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)
    vae.eval()

    prior = LowPrior(3, args.context_len // 2)
    if os.path.exists(BOTTOM_PRIOR_PATH):
        prior.load_state_dict(torch.load(BOTTOM_PRIOR_PATH, map_location='cpu'))
    prior.to(device)

    batch = next(load_text_samples(args.data, 1, args.context_len))
    batch = batch.to(device)

    all_latents = []
    inputs = batch
    for encoder in vae.encoders:
        encoded = encoder.encode(inputs)
        inputs, _, latents = encoder.vq(encoded)
        all_latents.append(latents)
    all_latents = all_latents[1:][::-1]

    context_len = args.context_len // 2
    latents = np.zeros([1, context_len], dtype=np.int64)
    for i in range(context_len):
        inputs = all_latents + [torch.from_numpy(latents).to(device)]
        outs = prior(*inputs)
        probs = torch.softmax(outs, dim=1)
        latents[:, i] = [sample_softmax(x) for x in probs.detach().cpu().numpy()[:, :, i]]
    all_latents.append(torch.from_numpy(latents).to(device))

    embedded = [encoder.vq.embed(latents) for encoder, latents
                in zip(vae.encoders[::-1], all_latents)]
    decoded = vae.decoders[-1](embedded)
    print_bytes('Original', batch[0])
    print_bytes('Reconstruction', torch.argmax(decoded[0], dim=-1))


if __name__ == '__main__':
    main()
