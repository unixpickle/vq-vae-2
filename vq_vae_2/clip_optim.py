"""
Utilities for more robust training.
"""

import math

import torch.optim as optim


class ClipOptim(optim.Adam):
    """
    An Adam optimizer that clips updates so that nothing
    catastrophic occurs.

    Args:
      params: learnable parameters to optimize.
      clip_frac: the fraction of updates to clip. This
        gives a sort of maximum update size threshold.
      hist_size: the number of previous update magnitudes
        to store when checking whether or not to clip.
        This also serves as the minimum number of steps to
        take before the updates are clipped at all.
      kwargs: arguments for Adam.
    """

    def __init__(self, params, clip_frac=0.05, hist_size=100, **kwargs):
        super().__init__(params, **kwargs)
        self.clip_frac = clip_frac
        self.hist_size = hist_size
        self._history = []

    def step(self, **kwargs):
        params = self._params_with_grad()
        old_params = [p.detach().clone() for p in params]
        super().step(**kwargs)
        total_norm = 0.0
        for cur, old in zip(params, old_params):
            total_norm += (cur.data - old).norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        clip_mag = self._magnitude_to_clip()
        self._history.append(total_norm)
        self._history = self._history[-self.hist_size:]
        if total_norm > clip_mag:
            _scale_update(old_params, params, clip_mag / total_norm)

    def _params_with_grad(self):
        return [p for group in self.param_groups for p in group['params'] if p.grad is not None]

    def _magnitude_to_clip(self):
        if len(self._history) < self.hist_size:
            return math.inf
        return sorted(self._history)[math.round(self.hist_size * (1 - self.clip_frac))]


def _scale_update(old_params, params, scale):
    for cur, old in zip(params, old_params):
        cur.data.copy_(old + scale * (cur.data - old))
