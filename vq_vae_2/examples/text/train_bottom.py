"""
Train the bottom-level prior.
"""

from vq_vae_2.examples.text.train_middle import main

BOTTOM_PRIOR_PATH = 'bottom.pt'

if __name__ == '__main__':
    main(prior_path=BOTTOM_PRIOR_PATH, num_levels=3)
