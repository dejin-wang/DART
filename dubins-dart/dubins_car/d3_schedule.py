# -*- coding: utf-8 -*-
"""D3 METRA -> Norm-Matching reward curriculum (schedule only).

D3 (Divide, Discover, Deploy) mixes alignment and norm-matching rewards:

    r = (1 - alpha) * r_METRA + alpha * r_NM

where alpha rises with the cosine similarity between Delta-phi and z.
Default switch band is [0.5, 0.7] (D3 Tab. 8 / metra.py).

On Dubins we take phi(s) = (x, y), so Delta-phi = u_{t+1} - u_t, and use the
same telescoping NM reward already in this repo (weight lambda). No factorized
skills, style priors, or encoder training are required.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F


class D3MixSchedule:
    """Tracks cosine(Delta-u, z) and exposes NM mix weight alpha in [0, 1]."""

    def __init__(
        self,
        switch_lo: float = 0.5,
        switch_hi: float = 0.7,
        window: int = 100,
        device=None,
    ):
        if switch_hi <= switch_lo:
            raise ValueError("switch_hi must be > switch_lo")
        self.switch_lo = float(switch_lo)
        self.switch_hi = float(switch_hi)
        self.window = int(window)
        self.device = device
        self._cos = deque(maxlen=self.window)
        self.alpha = 0.0  # weight on NM; 0 = pure METRA

    def observe(self, u_old: torch.Tensor, u_new: torch.Tensor, z: torch.Tensor) -> dict:
        delta = u_new - u_old
        cos = F.cosine_similarity(delta, z, dim=-1, eps=1e-8)
        mean_cos = float(cos.mean().detach())
        self._cos.append(mean_cos)
        avg = sum(self._cos) / len(self._cos)
        if avg <= self.switch_lo:
            self.alpha = 0.0
        elif avg >= self.switch_hi:
            self.alpha = 1.0
        else:
            self.alpha = (avg - self.switch_lo) / (self.switch_hi - self.switch_lo)
        return {
            "mean_cos": mean_cos,
            "ema_cos": avg,
            "alpha": self.alpha,
            "n": len(self._cos),
        }

    def snapshot(self) -> dict:
        return {
            "alpha": self.alpha,
            "ema_cos": (sum(self._cos) / len(self._cos)) if self._cos else 0.0,
            "switch_lo": self.switch_lo,
            "switch_hi": self.switch_hi,
            "n_obs": len(self._cos),
        }
