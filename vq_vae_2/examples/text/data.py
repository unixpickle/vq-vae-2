"""
Loading text datasets.
"""

import os
import random
import torch

import numpy as np


def load_text_samples(path, batch_size, context_len):
    """
    Load batches of snippets from a text file and yield
    the resulting [batch_size x context_len] Tensors of
    Long values, where each value stores a byte.
    """
    seqs = _load_individual_samples(path, context_len)
    while True:
        batch = np.zeros([batch_size, context_len], dtype=np.long)
        for i in range(batch_size):
            batch[i] = list(next(seqs))
        yield torch.from_numpy(batch)


def _load_individual_samples(path, context_len):
    size = os.path.getsize(path)
    with open(path, 'rb') as in_file:
        while True:
            in_file.seek(random.randrange(size - context_len), os.SEEK_SET)
            data = in_file.read(context_len)
            while len(data) < context_len:
                data += in_file.read(context_len - len(data))
            yield data
