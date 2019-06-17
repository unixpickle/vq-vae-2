import numpy as np
import torch

from .vq import embedding_distances


def test_embedding_distances():
    dictionary = torch.randn(15, 7)
    tensor = torch.randn(25, 13, 7)
    with torch.no_grad():
        actual = embedding_distances(dictionary, tensor).numpy()
        expected = naive_embedding_distances(dictionary, tensor).numpy()
        assert np.allclose(actual, expected, atol=1e-4)


def naive_embedding_distances(dictionary, tensor):
    return torch.sum(torch.pow(tensor[..., None, :] - dictionary, 2), dim=-1)
