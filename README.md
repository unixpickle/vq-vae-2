# vq-vae-2

This is a PyTorch implementation of [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446), including their PixelCNN and self-attention priors.

# To-do list

This is a work-in-progress. Here's my checklist:

  * [x] Implement Gated PixelCNN with conditioning
  * [x] Implement masked self-attention
  * [x] Test PixelCNN on MNIST
  * [x] Implement vector quantizing layer
  * [x] Implement VQ-VAE encoder/decoder
  * [x] Test VQ-VAE + PixelCNN on MNIST
  * [x] Implement hierarchical VQ-VAE
  * [x] Train hierarchical VQ-VAE on large images
  * [x] Train top PixelCNN on large images
  * [ ] Train bottom PixelCNN on large images
