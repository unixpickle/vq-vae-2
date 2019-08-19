"""
Produce samples from all the priors combined.
"""

import os

import numpy as np
import torch

from vq_vae_2.examples.text.model import LowPrior, TopPrior, make_vae
from vq_vae_2.examples.text.recon_vae import print_bytes
from vq_vae_2.examples.text.sample_top import arg_parser, sample_softmax
from vq_vae_2.examples.text.train_bottom import BOTTOM_PRIOR_PATH
from vq_vae_2.examples.text.train_middle import MIDDLE_PRIOR_PATH
from vq_vae_2.examples.text.train_top import TOP_PRIOR_PATH
from vq_vae_2.examples.text.train_vae import VAE_PATH


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)

    vae = make_vae()
    if os.path.exists(VAE_PATH):
        vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae.to(device)
    vae.eval()

    top_prior = TopPrior(args.context_len // 8)
    if os.path.exists(TOP_PRIOR_PATH):
        top_prior.load_state_dict(torch.load(TOP_PRIOR_PATH, map_location='cpu'))
    top_prior.to(device)

    priors = [top_prior]

    for i, path in enumerate([MIDDLE_PRIOR_PATH, BOTTOM_PRIOR_PATH]):
        prior = LowPrior(i + 2, args.context_len >> (2 - i))
        if os.path.exists(path):
            prior.load_state_dict(torch.load(path, map_location='cpu'))
        prior.to(device)
        priors.append(prior)

    all_latents = []
    for i, prior in enumerate(priors):
        context_len = args.context_len >> (3 - i)
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
    print_bytes('Reconstruction', torch.argmax(decoded[0], dim=-1))


if __name__ == '__main__':
    main()
