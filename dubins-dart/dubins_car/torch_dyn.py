# -*- coding: utf-8 -*-
"""Torch copies of the Dubins kinematics. Kept separate so the Gym env can import without torch."""

from __future__ import annotations

import math

import torch

from dubins_car.dynamics import DT, OMEGA_MAX, V


def step_dynamics_torch(state, action, dt=DT, v=V, omega_max=OMEGA_MAX):
    x, y, th = state[:, 0], state[:, 1], state[:, 2]
    omega = action.reshape(-1).clamp(-1.0, 1.0) * omega_max
    half = 0.5 * omega * dt
    sinc = torch.sinc(half / math.pi)
    x = x + v * dt * sinc * torch.cos(th + half)
    y = y + v * dt * sinc * torch.sin(th + half)
    th = th + omega * dt
    return torch.stack([x, y, th], dim=-1)


def rollout_actions_torch(actions, dt=DT, v=V, omega_max=OMEGA_MAX):
    """actions: (B, H) in [-1, 1]. Returns xy (B, 2) and final heading (B,)."""
    omega = actions.clamp(-1.0, 1.0) * omega_max
    dth = omega * dt
    th_end = torch.cumsum(dth, dim=-1)
    th_start = th_end - dth
    half = 0.5 * dth
    sinc = torch.sinc(half / math.pi)
    dx = v * dt * sinc * torch.cos(th_start + half)
    dy = v * dt * sinc * torch.sin(th_start + half)
    xy = torch.stack([dx.sum(-1), dy.sum(-1)], dim=-1)
    return xy, th_end[..., -1]


def make_obs_torch(state, z, u, t, horizon, path_length):
    th = state[:, 2]
    tt = (t.to(dtype=state.dtype) / float(horizon)).unsqueeze(-1)
    return torch.cat(
        [
            torch.cos(th).unsqueeze(-1),
            torch.sin(th).unsqueeze(-1),
            z,
            u / path_length,
            tt,
        ],
        dim=-1,
    )


def _as_weight(weight, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(weight):
        return weight.to(device=like.device, dtype=like.dtype).reshape(-1)
    return torch.full((like.shape[0],), float(weight), device=like.device, dtype=like.dtype)


def reward_torch(u_old, u_new, z, mode: str, weight, alpha=0.0):
    fwd = ((u_new - u_old) * z).sum(-1)
    b_old = (-z[:, 1] * u_old[:, 0] + z[:, 0] * u_old[:, 1]).abs()
    b_new = (-z[:, 1] * u_new[:, 0] + z[:, 0] * u_new[:, 1]).abs()
    w = _as_weight(weight, fwd)
    if mode == "metra":
        return fwd
    if mode in ("dart", "adaptive"):
        return fwd - w * (b_new - b_old)
    if mode == "squared":
        return fwd - w * (b_new.square() - b_old.square())
    if mode == "norm":
        return (1.0 + w) * fwd - w * (u_new.norm(dim=-1) - u_old.norm(dim=-1))
    if mode == "d3":
        # D3 schedule: (1-alpha) METRA + alpha NM, alpha in [0,1].
        a = _as_weight(alpha, fwd)
        r_m = fwd
        r_n = (1.0 + w) * fwd - w * (u_new.norm(dim=-1) - u_old.norm(dim=-1))
        return (1.0 - a) * r_m + a * r_n
    raise ValueError(mode)


def final_score_torch(u, z, mode: str, weight: float):
    a = (u * z).sum(-1)
    b = (-z[:, 1] * u[:, 0] + z[:, 0] * u[:, 1]).abs()
    if mode == "metra":
        return a
    if mode == "dart":
        return a - float(weight) * b
    if mode == "squared":
        return a - float(weight) * b.square()
    if mode == "norm":
        return (1.0 + float(weight)) * a - float(weight) * u.norm(dim=-1)
    if mode == "radial":
        lat = -z[:, 1] * u[:, 0] + z[:, 0] * u[:, 1]
        return a - float(weight) * lat.square()
    raise ValueError(mode)


def score_matrix(xy, z, mode: str, weight: float):
    """xy (N, 2), z (K, 2) -> scores (N, K)."""
    a = xy @ z.T
    zp = torch.stack([-z[:, 1], z[:, 0]], dim=-1)
    lat = xy @ zp.T
    b = lat.abs()
    if mode in ("metra", "dart"):
        w = 0.0 if mode == "metra" else float(weight)
        return a - w * b
    if mode == "squared":
        return a - float(weight) * b.square()
    if mode == "norm":
        return (1.0 + float(weight)) * a - float(weight) * xy.norm(dim=-1, keepdim=True)
    if mode == "radial":
        return a - float(weight) * lat.square()
    raise ValueError(mode)
