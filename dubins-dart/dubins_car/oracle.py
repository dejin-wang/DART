# -*- coding: utf-8 -*-
"""Ideal skill endpoints for the fixed-horizon Dubins car.

Controls are sequences of the same length and step size as the Gym environment,
so the maximizer is the ideal policy for that discrete car, not a continuous
relaxation.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from dubins_car.dynamics import DT, OMEGA_MAX, V, horizon_steps
from dubins_car.metrics import alignment_bound_rad, angular_gap, direction_errors
from dubins_car.torch_dyn import final_score_torch, rollout_actions_torch, score_matrix


def skill_grid(k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    psi = torch.arange(k, device=device, dtype=torch.float32) * (2 * math.pi / k)
    z = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)
    return psi, z


def build_candidates(horizon: int, seed: int = 0) -> np.ndarray:
    chunks = []
    for sgn1 in (-1.0, 1.0):
        for sgn2 in (-1.0, 1.0):
            for s in range(horizon + 1):
                a = np.full(horizon, sgn2, dtype=np.float32)
                a[:s] = sgn1
                chunks.append(a)
    stride = 4
    grid = range(0, horizon + 1, stride)
    for s1 in grid:
        for s2 in grid:
            if s2 < s1:
                continue
            for sgn1 in (-1.0, 1.0):
                for sgn2 in (-1.0, 1.0):
                    a = np.zeros(horizon, dtype=np.float32)
                    if s1:
                        a[:s1] = sgn1
                    if s2 < horizon:
                        a[s2:] = sgn2
                    chunks.append(a)
    consts = np.linspace(-1.0, 1.0, 41, dtype=np.float32)[:, None] * np.ones((1, horizon), dtype=np.float32)
    rng = np.random.default_rng(seed)
    rand = rng.uniform(-1.0, 1.0, size=(6000, horizon)).astype(np.float32)
    bang = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), size=(4000, horizon))
    return np.concatenate([np.stack(chunks, 0), consts, rand, bang], 0)


def turn_then_straight(psi: torch.Tensor, horizon: int, dt: float, long_way: bool) -> torch.Tensor:
    ang = torch.atan2(torch.sin(psi), torch.cos(psi))
    if long_way:
        mag = 2 * math.pi - ang.abs()
        sgn = torch.where(ang > 0, -torch.ones_like(ang), torch.ones_like(ang))
        sgn = torch.where(ang.abs() < 1e-6, torch.ones_like(ang), sgn)
    else:
        mag = ang.abs()
        sgn = torch.where(ang >= 0, torch.ones_like(ang), -torch.ones_like(ang))
    turn_time = mag / OMEGA_MAX
    t = torch.arange(horizon, device=psi.device, dtype=psi.dtype) * dt
    turning = t.unsqueeze(0) < turn_time.unsqueeze(1)
    return torch.where(turning, sgn.unsqueeze(1).expand_as(turning), torch.zeros_like(turning, dtype=psi.dtype))


def polish(actions, z, mode, weight, dt, steps=400, lr=0.03):
    raw = actions.detach().clamp(-1, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    best_omega = raw.detach().clone()
    with torch.no_grad():
        u0, _ = rollout_actions_torch(best_omega, dt=dt)
        best_score = final_score_torch(u0, z, mode, weight)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        u, _ = rollout_actions_torch(raw, dt=dt)
        score = final_score_torch(u, z, mode, weight)
        (-score.mean()).backward()
        with torch.no_grad():
            s = score.detach()
            improve = s > best_score
            best_score = torch.where(improve, s, best_score)
            best_omega = torch.where(improve.unsqueeze(-1), raw.detach(), best_omega)
        opt.step()
        with torch.no_grad():
            raw.clamp_(-1, 1)
    with torch.no_grad():
        u, _ = rollout_actions_torch(best_omega, dt=dt)
        score = final_score_torch(u, z, mode, weight)
        improve = score > best_score
        best_score = torch.where(improve, score, best_score)
        best_omega = torch.where(improve.unsqueeze(-1), best_omega, best_omega)
        u, _ = rollout_actions_torch(best_omega, dt=dt)
    return best_omega.detach(), u.detach(), best_score.detach()


def _select_and_polish(cand_a, cand_xy, z, mode, weight, heuristics, dt, steps):
    scores = score_matrix(cand_xy, z, mode, weight)
    idx = scores.argmax(dim=0)
    inits = [cand_a[idx], *heuristics]
    init = torch.cat(inits, dim=0)
    z_rep = z.repeat(len(inits), 1)
    _, u, sc = polish(init, z_rep, mode, weight, dt, steps=steps)
    k = z.shape[0]
    sc = sc.view(len(inits), k)
    u = u.view(len(inits), k, 2)
    pick = sc.argmax(dim=0)
    chosen = u[pick, torch.arange(k, device=z.device)]
    return chosen, sc.max(dim=0).values


def circular_derivative(y: np.ndarray, dpsi: float) -> np.ndarray:
    pad = np.concatenate([[y[-1]], y, [y[0]]])
    return (pad[2:] - pad[:-2]) / (2 * dpsi)


def smooth_circular(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (x / sigma_bins) ** 2)
    ker /= ker.sum()
    pad = np.pad(y, radius, mode="wrap")
    return np.convolve(pad, ker, mode="valid")


def _pack_result(z_np, u_np, mode, weight, L):
    err = direction_errors(z_np, u_np)
    gap = angular_gap(u_np)
    bound = alignment_bound_rad(L, weight) if mode == "dart" else float("nan")
    return {
        "mode": mode,
        "weight": float(weight),
        "max_err_deg": float(np.degrees(err.max())),
        "mean_err_deg": float(np.degrees(err.mean())),
        "median_err_deg": float(np.degrees(np.median(err))),
        "blind_deg": gap["gap_deg"],
        "blind_excess_deg": gap["excess_deg"],
        "blind_center_deg": gap["center_deg"],
        "blind_start_deg": gap["start_deg"],
        "bound_deg": float(np.degrees(bound)) if bound == bound else float("nan"),
        "mean_forward": float(np.sum(z_np * u_np, axis=1).mean()),
        "mean_norm": float(np.linalg.norm(u_np, axis=1).mean()),
        "u": u_np,
        "err": err,
    }


def ideal_skills_for_horizon(
    horizon_radii: float,
    dart_weights,
    squared_weights,
    norm_weights,
    device,
    k: int = 360,
    seed: int = 0,
    polish_steps: int = 350,
):
    h = horizon_steps(horizon_radii)
    psi_t, z = skill_grid(k, device)
    cand_np = build_candidates(h, seed=seed)
    cand_a = torch.from_numpy(cand_np).to(device)
    with torch.no_grad():
        cand_xy, _ = rollout_actions_torch(cand_a, dt=DT)
    short = turn_then_straight(psi_t, h, DT, long_way=False)
    long = turn_then_straight(psi_t, h, DT, long_way=True)
    heuristics = [short, long]

    # Radial boundary rho(psi): farthest point on each ray.
    u_rad, _ = _select_and_polish(cand_a, cand_xy, z, "radial", 800.0, heuristics, DT, polish_steps)
    lat = (-z[:, 1] * u_rad[:, 0] + z[:, 0] * u_rad[:, 1]).abs()
    if float(lat.max()) > 0.08:
        _, u2, _ = polish(short, z, "radial", 8000.0, DT, steps=polish_steps)
        lat2 = (-z[:, 1] * u2[:, 0] + z[:, 0] * u2[:, 1]).abs()
        use2 = lat2 < lat
        u_rad = torch.where(use2.unsqueeze(-1), u2, u_rad)
        lat = torch.where(use2, lat2, lat)

    along = (u_rad * z).sum(-1)
    with torch.no_grad():
        ang = torch.atan2(cand_xy[:, 1], cand_xy[:, 0])
        dang = torch.atan2(torch.sin(ang[:, None] - psi_t[None, :]), torch.cos(ang[:, None] - psi_t[None, :])).abs()
        proj = cand_xy @ z.T
        proj = proj.masked_fill(dang > (1.0 * math.pi / 180.0), -1e9)
        bin_max = proj.max(dim=0).values
    rho_t = torch.maximum(along, torch.zeros_like(along))
    # Prefer the optimized point when it actually lies on the ray.
    rho_t = torch.where(lat < 0.1, torch.clamp(along, min=0.0), torch.clamp(bin_max, min=0.0))
    rho = rho_t.detach().cpu().numpy()
    psi = psi_t.detach().cpu().numpy()
    dpsi = float(psi[1] - psi[0])
    log_rho = np.log(np.clip(rho, 1e-4, None))
    d_raw = circular_derivative(log_rho, dpsi)
    d_smooth = circular_derivative(smooth_circular(log_rho, 2.0), dpsi)
    finite = np.isfinite(d_raw) & (rho > 0.2)
    L_raw = float(np.max(np.abs(d_raw[finite]))) if finite.any() else float("nan")
    L = float(np.max(np.abs(d_smooth[np.isfinite(d_smooth)])))

    cloud = cand_xy[:: max(1, cand_xy.shape[0] // 4000)].detach().cpu().numpy()

    # Straight-ahead diagnostic: METRA on +x must travel essentially distance H*dt.
    with torch.no_grad():
        z_east = torch.tensor([[1.0, 0.0]], device=device)
        east_score = float(score_matrix(cand_xy, z_east, "metra", 0.0).max())

    specs = [("dart", float(w)) for w in dart_weights]
    specs += [("squared", float(w)) for w in squared_weights]
    specs += [("norm", float(w)) for w in norm_weights]
    results = []
    z_np = z.detach().cpu().numpy()
    for mode, weight in specs:
        u, _ = _select_and_polish(cand_a, cand_xy, z, mode, weight, heuristics, DT, polish_steps)
        u_np = u.detach().cpu().numpy()
        packed = _pack_result(z_np, u_np, mode, weight, L)
        results.append(packed)
        print(
            f"  H={horizon_radii:g} {mode} w={weight:g}  "
            f"max_err={packed['max_err_deg']:.2f} deg  mean={packed['mean_err_deg']:.2f}  "
            f"blind={packed['blind_deg']:.2f} (center {packed['blind_center_deg']:.1f})  "
            f"bound={packed['bound_deg']:.2f}",
            flush=True,
        )

    # Star-shaped check: can we stop halfway along the outer ray at the fixed horizon?
    check_idx = torch.arange(0, k, max(1, k // 24), device=device)
    targets = (0.55 * rho_t[check_idx]).unsqueeze(-1) * z[check_idx]
    init = turn_then_straight(psi_t[check_idx], h, DT, long_way=False)
    raw = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(220):
        opt.zero_grad(set_to_none=True)
        u, _ = rollout_actions_torch(raw, dt=DT)
        loss = ((u - targets) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            raw.clamp_(-1, 1)
    with torch.no_grad():
        u_fit, _ = rollout_actions_torch(raw.detach(), dt=DT)
        fit_err = (u_fit - targets).norm(dim=-1)
    star_median = float(fit_err.median())
    star_frac = float((fit_err < 0.15).float().mean())

    print(
        f"H={horizon_radii:g} steps={h}  L_smooth={L:.3f}  L_raw={L_raw:.3f}  "
        f"rho=[{rho.min():.2f}, {rho.max():.2f}]  east_proj={east_score:.3f}  "
        f"star_median_err={star_median:.3f}  star_frac@0.15={star_frac:.2f}  "
        f"radial_lat_max={float(lat.max()):.3f}",
        flush=True,
    )
    if east_score < 0.98 * h * DT * V:
        raise RuntimeError(f"oracle failed the straight-line check: east projection {east_score}")

    return {
        "horizon_radii": float(horizon_radii),
        "steps": h,
        "psi": psi,
        "rho": rho,
        "dlog_raw": d_raw,
        "dlog_smooth": d_smooth,
        "L": L,
        "L_raw": L_raw,
        "z": z_np,
        "cloud": cloud,
        "results": results,
        "east_projection": east_score,
        "star_median_err": star_median,
        "star_frac": star_frac,
        "radial_lat_max": float(lat.max()),
    }
