"""
Sample an image from a PixelCNN.
"""

import random

from PIL import Image
import numpy as np
import torch

from vq_vae_2.examples.mnist.model import Model


def main():
    model = Model()
    model.load_state_dict(torch.load('pixel_cnn.pt'))
    inputs = np.zeros([1, 1, 28, 28], dtype=np.float32)
    for row in range(28):
        for col in range(28):
            with torch.no_grad():
                outputs = torch.sigmoid(model(torch.from_numpy(inputs))).numpy()
                if random.random() < outputs[0, 0, row, col]:
                    inputs[0, 0, row, col] = 1.0
        print('done row', row)
    Image.fromarray((inputs * 255).astype(np.uint8)[0, 0]).save('digit.png')


if __name__ == '__main__':
    main()
