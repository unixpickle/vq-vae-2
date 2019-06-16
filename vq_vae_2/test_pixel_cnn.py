import pytest

import numpy as np
import torch
import torch.nn as nn

from .attention import PixelAttention
from .pixel_cnn import PixelCNN, PixelConvA, PixelConvB

TEST_IMG_WIDTH = 7
TEST_IMG_HEIGHT = 11
TEST_IMG_DEPTH_IN = 2
TEST_IMG_DEPTH = 4


@pytest.mark.parametrize('start,middle', [
    (
        PixelConvA(TEST_IMG_DEPTH_IN, TEST_IMG_DEPTH, horizontal=2, vertical=3),
        PixelAttention(TEST_IMG_DEPTH, num_heads=2),
    ),
    (
        PixelConvA(TEST_IMG_DEPTH_IN, TEST_IMG_DEPTH, horizontal=2, vertical=2),
        PixelConvB(TEST_IMG_DEPTH, horizontal=2, vertical=2),
    ),
    (
        PixelConvA(TEST_IMG_DEPTH_IN, TEST_IMG_DEPTH, horizontal=3, vertical=2),
        PixelConvB(TEST_IMG_DEPTH, horizontal=3, vertical=2),
    ),
    (
        PixelConvA(TEST_IMG_DEPTH_IN, TEST_IMG_DEPTH, horizontal=2, vertical=3),
        PixelConvB(TEST_IMG_DEPTH, horizontal=2, vertical=3),
    ),
])
def test_pixel_cnn_masking(start, middle):
    network = PixelCNN(start, middle)
    outer_idx = 0
    for row in range(TEST_IMG_HEIGHT):
        for col in range(TEST_IMG_WIDTH):
            for z in range(TEST_IMG_DEPTH):
                input_img = nn.Parameter(torch.randn(1, TEST_IMG_DEPTH_IN, TEST_IMG_HEIGHT,
                                                     TEST_IMG_WIDTH))
                output = network(input_img)
                output = output[0][0, z, row, col] + 0.9 * output[1][0, z, row, col]
                output.backward()
                gradient = input_img.grad.data.numpy()
                if outer_idx > 0:
                    assert np.max(np.abs(gradient)) > 1e-4, 'at %d,%d,%d' % (row, col, z)
                inner_idx = 0
                for inner_row in range(TEST_IMG_HEIGHT):
                    for inner_col in range(TEST_IMG_WIDTH):
                        for inner_z in range(TEST_IMG_DEPTH_IN):
                            should_be_zero = inner_idx >= outer_idx
                            grad_val = gradient[0, inner_z, inner_row, inner_col]
                            if should_be_zero:
                                assert abs(grad_val) < 1e-5
                        inner_idx += 1
            outer_idx += 1
