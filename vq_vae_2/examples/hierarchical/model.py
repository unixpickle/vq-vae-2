"""
Models for hierarchical image generation.
"""

from vq_vae_2.vq_vae import HalfDecoder, HalfQuarterDecoder, HalfEncoder, QuarterEncoder, VQVAE


def make_vae():
    encoders = [QuarterEncoder(3, 128, 512), HalfEncoder(128, 128, 512)]
    decoders = [HalfDecoder(128, 128), HalfQuarterDecoder(128, 3)]
    return VQVAE(encoders, decoders)
