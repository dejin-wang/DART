# -*- coding: utf-8 -*-
"""Direction error and blind-sector measurements shared by the oracle and SAC."""

from __future__ import annotations

import math

import numpy as np


def direction_errors(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Angle in radians between each commanded code and its displacement."""
    nu = np.linalg.norm(u, axis=-1)
    c = np.sum(z * u, axis=-1) / np.clip(nu, 1e-8, None)
    return np.arccos(np.clip(c, -1.0, 1.0))


def angular_gap(u: np.ndarray, min_norm: float = 0.2) -> dict:
    """Largest hole in the set of endpoint directions.

    A uniform grid of K endpoints has a gap of about 360/K degrees. Anything
    much larger is the blind sector: directions no selected skill ends in.
    """
    nu = np.linalg.norm(u, axis=-1)
    keep = nu > min_norm
    empty = {
        "gap_deg": 360.0,
        "excess_deg": 360.0,
        "center_deg": 180.0,
        "start_deg": 0.0,
        "n": 0,
    }
    if int(keep.sum()) < 8:
        return empty
    ang = np.arctan2(u[keep, 1], u[keep, 0]) % (2 * math.pi)
    ang = np.sort(ang)
    diffs = np.diff(ang, append=ang[0] + 2 * math.pi)
    i = int(np.argmax(diffs))
    width = float(diffs[i])
    start = float(ang[i])
    center = (start + 0.5 * width) % (2 * math.pi)
    uniform = 2 * math.pi / len(ang)
    return {
        "gap_deg": width * 180.0 / math.pi,
        "excess_deg": max(0.0, (width - uniform) * 180.0 / math.pi),
        "center_deg": center * 180.0 / math.pi,
        "start_deg": start * 180.0 / math.pi,
        "n": int(len(ang)),
    }


def alignment_bound_rad(L: float, beta: float) -> float:
    """Theorem bound: angle <= [arctan(L) - arctan(beta)]_+."""
    return max(0.0, math.atan(L) - math.atan(beta))
