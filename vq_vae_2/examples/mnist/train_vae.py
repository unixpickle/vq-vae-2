"""
Train an encoder/decoder on the MNIST dataset.
"""

import os

from PIL import Image
import numpy as np
import torch
import torch.optim as optim

from vq_vae_2.examples.mnist.model import Encoder, Decoder
from vq_vae_2.examples.mnist.train import load_images
from vq_vae_2.vq import vq_loss


def main():
    enc = Encoder()
    dec = Decoder()
    if os.path.exists('enc.pt'):
        enc.load_state_dict(torch.load('enc.pt'))
    if os.path.exists('dec.pt'):
        dec.load_state_dict(torch.load('dec.pt'))

    optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()))
    enc.train()
    for i, batch in enumerate(load_images()):
        raw_enc, (embedded, embedded_pt, _) = enc(batch)
        decoded = dec(embedded_pt)
        mse_loss = torch.mean(torch.pow(decoded - batch, 2))
        loss = mse_loss + vq_loss(raw_enc, embedded)
        print('step %d: loss=%f mse=%f' % (i, loss.item(), mse_loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not i % 10:
            torch.save(enc.state_dict(), 'enc.pt')
            torch.save(dec.state_dict(), 'dec.pt')
        if not i % 100:
            save_reconstructions(batch, decoded)


def save_reconstructions(batch, decoded):
    batch = batch.detach().permute(0, 2, 3, 1).contiguous()
    decoded = decoded.detach().permute(0, 2, 3, 1).contiguous()
    input_images = (np.concatenate(batch.numpy(), axis=0) * 255).astype(np.uint8)
    output_images = np.concatenate(decoded.numpy(), axis=0)
    output_images = (np.clip(output_images, 0, 1) * 255).astype(np.uint8)
    joined = np.concatenate([input_images[..., 0], output_images[..., 0]], axis=1)
    Image.fromarray(joined).save('reconstructions.png')


if __name__ == '__main__':
    main()
