# -*- coding: utf-8 -*-
"""Dubins car kinematics and DART / METRA rewards.

The car always drives forward at speed v with turning rate |omega| <= omega_max.
phi(s) = (x, y) is fixed. One environment step uses the exact constant-curvature
displacement, so a straight skill of H steps travels exactly H * v * dt.
"""

from __future__ import annotations

import math

import numpy as np

DT = 0.05
V = 1.0
OMEGA_MAX = 1.0
R_MIN = V / OMEGA_MAX
OBS_DIM = 7


def horizon_steps(horizon_radii: float, dt: float = DT, v: float = V, omega_max: float = OMEGA_MAX) -> int:
    """Number of steps whose path length equals horizon_radii turning radii."""
    length = float(horizon_radii) * (v / omega_max)
    steps = int(round(length / (v * dt)))
    if abs(steps * v * dt - length) > 1e-8:
        raise ValueError(f"horizon {horizon_radii} is not an integer number of steps at dt={dt}")
    return steps


def integrate_step_np(state, action, dt=DT, v=V, omega_max=OMEGA_MAX):
    x, y, th = (float(state[0]), float(state[1]), float(state[2]))
    omega = float(np.clip(action, -1.0, 1.0)) * omega_max
    half = 0.5 * omega * dt
    sinc = float(np.sinc(half / np.pi))  # sin(half) / half
    x = x + v * dt * sinc * math.cos(th + half)
    y = y + v * dt * sinc * math.sin(th + half)
    th = th + omega * dt
    return np.array([x, y, th], dtype=np.float64)


def _lateral(u, z):
    # z_perp = (-z_y, z_x), b = |z_perp · u|
    return abs(float(-z[1] * u[0] + z[0] * u[1]))


def reward_np(u_old, u_new, z, mode: str, weight: float, alpha: float = 0.0) -> float:
    """One-step reward. The H-step sum equals the terminal objective F(u_H)."""
    u_old = np.asarray(u_old, dtype=np.float64)
    u_new = np.asarray(u_new, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    fwd = float((u_new - u_old) @ z)
    b_old = _lateral(u_old, z)
    b_new = _lateral(u_new, z)
    if mode == "metra":
        return fwd
    if mode == "dart":
        return fwd - float(weight) * (b_new - b_old)
    if mode == "squared":
        return fwd - float(weight) * (b_new * b_new - b_old * b_old)
    if mode == "norm":
        n_old = float(np.linalg.norm(u_old))
        n_new = float(np.linalg.norm(u_new))
        return (1.0 + float(weight)) * fwd - float(weight) * (n_new - n_old)
    if mode == "d3":
        a = float(alpha)
        r_m = fwd
        n_old = float(np.linalg.norm(u_old))
        n_new = float(np.linalg.norm(u_new))
        r_n = (1.0 + float(weight)) * fwd - float(weight) * (n_new - n_old)
        return (1.0 - a) * r_m + a * r_n
    raise ValueError(mode)


def terminal_objective_np(u, z, mode: str, weight: float) -> float:
    u = np.asarray(u, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    a = float(u @ z)
    b = _lateral(u, z)
    if mode == "metra":
        return a
    if mode == "dart":
        return a - float(weight) * b
    if mode == "squared":
        return a - float(weight) * b * b
    if mode == "norm":
        return (1.0 + float(weight)) * a - float(weight) * float(np.linalg.norm(u))
    raise ValueError(mode)


def make_obs_np(state, z, u, t, horizon, path_length) -> np.ndarray:
    th = float(state[2])
    return np.array(
        [
            math.cos(th),
            math.sin(th),
            float(z[0]),
            float(z[1]),
            float(u[0]) / path_length,
            float(u[1]) / path_length,
            float(t) / float(horizon),
        ],
        dtype=np.float32,
    )


def self_test() -> None:
    # Quarter circle of radius 1 ends at (1, 1) with heading pi/2.
    s = integrate_step_np(np.zeros(3), 1.0, dt=math.pi / 2)
    assert abs(s[0] - 1.0) < 1e-8 and abs(s[1] - 1.0) < 1e-8
    assert abs(s[2] - math.pi / 2) < 1e-8
    # Full circle returns to the origin.
    s = integrate_step_np(np.zeros(3), 1.0, dt=2 * math.pi)
    assert abs(s[0]) < 1e-8 and abs(s[1]) < 1e-8
    # Straight line travels exactly the elapsed time.
    s = np.zeros(3)
    for _ in range(200):
        s = integrate_step_np(s, 0.0, dt=DT)
    assert abs(s[0] - 10.0) < 1e-8 and abs(s[1]) < 1e-8

    rng = np.random.default_rng(0)
    z = np.array([0.6, 0.8])
    state = np.zeros(3)
    u = np.zeros(2)
    totals = {m: 0.0 for m in ("metra", "dart", "squared", "norm")}
    weights = {"metra": 0.0, "dart": 0.4, "squared": 0.2, "norm": 1.5}
    for _ in range(80):
        state = integrate_step_np(state, float(rng.uniform(-1, 1)))
        u_new = state[:2].copy()
        for mode, w in weights.items():
            totals[mode] += reward_np(u, u_new, z, mode, w)
        u = u_new
    for mode, w in weights.items():
        got = terminal_objective_np(u, z, mode, w)
        if abs(totals[mode] - got) > 1e-6:
            raise AssertionError(f"{mode} telescope {totals[mode]} != {got}")

    assert horizon_steps(10) == 200
    assert horizon_steps(6) == 120
    print("dynamics self-test ok", flush=True)
