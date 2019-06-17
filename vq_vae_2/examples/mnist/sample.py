"""
Sample an image from a PixelCNN.
"""

import random

from PIL import Image
import numpy as np
import torch

from vq_vae_2.examples.mnist.model import Encoder, Decoder, Generator


def main():
    encoder = Encoder()
    encoder.load_state_dict(torch.load('enc.pt'))
    encoder.eval()
    decoder = Decoder()
    decoder.load_state_dict(torch.load('dec.pt'))
    generator = Generator()
    generator.load_state_dict(torch.load('gen.pt'))

    inputs = np.zeros([1, 7, 7], dtype=np.long)
    for row in range(7):
        for col in range(7):
            with torch.no_grad():
                outputs = torch.softmax(generator(torch.from_numpy(inputs)), dim=1).numpy()
                probs = outputs[0, :, row, col]
                inputs[0, row, col] = sample_softmax(probs)
        print('done row', row)
    embedded = encoder.vq.embed(torch.from_numpy(inputs))
    decoded = torch.clamp(decoder(embedded), 0, 1).detach().numpy()
    Image.fromarray((decoded * 255).astype(np.uint8)[0, 0]).save('digit.png')


def sample_softmax(probs):
    number = random.random()
    for i, x in enumerate(probs):
        number -= x
        if number <= 0:
            return i
    return len(probs) - 1


if __name__ == '__main__':
    main()
