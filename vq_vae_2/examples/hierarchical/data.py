"""
Data loading pipeline.

All data should be stored as images in a single directory.
"""

import os
import random

from PIL import Image
import numpy as np
import torch

IMAGE_SIZE = 256


def load_images(dir_path, batch_size=16):
    images = load_single_images(dir_path)
    while True:
        batch = np.array([next(images) for _ in range(batch_size)])
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        batch = batch.float() / 255
        yield batch


def load_single_images(dir_path):
    while True:
        with os.scandir(dir_path) as listing:
            for entry in listing:
                if not (entry.name.endswith('.png') or entry.name.endswith('.jpg')):
                    continue
                try:
                    img = Image.open(entry.path)
                except OSError:
                    # Ignore corrupt images.
                    continue
                width, height = img.size
                scale = IMAGE_SIZE / min(width, height)
                img = img.resize((round(scale * width), round(scale * height)))
                img = img.convert('RGB')
                tensor = np.array(img)
                row = random.randrange(tensor.shape[0] - IMAGE_SIZE + 1)
                col = random.randrange(tensor.shape[1] - IMAGE_SIZE + 1)
                yield tensor[row:row + IMAGE_SIZE, col:col + IMAGE_SIZE]
