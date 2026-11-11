# -*- coding: utf-8 -*-
"""Direction-adaptive absolute lateral penalty (DART).

Legacy AdaptiveBeta:
  beta(psi) = beta_base * (1 + e_hat(psi))^eta / E[(1 + e_hat)^eta],
  with e = sin^2(theta/2). Mean beta is preserved before clipping.

Cone-integral AdaptiveConeAlpha (preferred):
  Maintain cone half-angle alpha(z) on S^1 bins and dual-ascend
    alpha <- Pi(alpha + kappa (hat_theta - theta_tol)),  beta = tan(alpha),
  where hat_theta = arctan(sum b / sum a) aggregates lateral/forward per bin.
  This is multiplier ascent for E[b] <= tan(theta_tol) E[a]. In d=2, equally
  spaced anchors with a box kernel reduce to angular bins.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def angular_error_from_u(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """e = 0.5 * (1 - cos theta) = sin^2(theta/2) in [0, 1]."""
    nu = u.norm(dim=-1).clamp_min(1e-8)
    cos = (z * u).sum(-1) / nu
    return (0.5 * (1.0 - cos.clamp(-1.0, 1.0))).clamp(0.0, 1.0)


def forward_lateral(z: torch.Tensor, u: torch.Tensor):
    """a = z·u, b = ||(I-zz^T)u|| (= |lat| in 2D)."""
    a = (z * u).sum(-1)
    b = (-z[:, 1] * u[:, 0] + z[:, 0] * u[:, 1]).abs()
    return a, b


def periodic_linear_interp(psi_query: np.ndarray, psi_obs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate a 2-pi-periodic scalar field with linear pieces."""
    if len(psi_obs) == 0:
        return np.zeros_like(psi_query, dtype=np.float64)
    order = np.argsort(psi_obs)
    psi = psi_obs[order] % (2 * math.pi)
    val = values[order].astype(np.float64)
    # Deduplicate nearly identical angles by averaging.
    keep_psi = [float(psi[0])]
    keep_val = [float(val[0])]
    counts = [1]
    for p, v in zip(psi[1:], val[1:]):
        if abs(p - keep_psi[-1]) < 1e-6:
            keep_val[-1] = (keep_val[-1] * counts[-1] + float(v)) / (counts[-1] + 1)
            counts[-1] += 1
        else:
            keep_psi.append(float(p))
            keep_val.append(float(v))
            counts.append(1)
    psi = np.asarray(keep_psi, dtype=np.float64)
    val = np.asarray(keep_val, dtype=np.float64)
    if len(psi) == 1:
        return np.full_like(psi_query, val[0], dtype=np.float64)

    psi_ext = np.concatenate([psi - 2 * math.pi, psi, psi + 2 * math.pi])
    val_ext = np.concatenate([val, val, val])
    q = np.asarray(psi_query, dtype=np.float64) % (2 * math.pi)
    return np.interp(q, psi_ext, val_ext)


class AdaptiveBeta:
    """Piecewise-linear beta schedule over skill angle."""

    def __init__(self, beta_base: float, n_bins: int = 72, device=None, eta: float = 1.0):
        self.beta_base = float(beta_base)
        self.n_bins = int(n_bins)
        self.device = device
        self.eta = float(eta)
        self.psi_grid = np.linspace(0.0, 2 * math.pi, self.n_bins, endpoint=False)
        self.e_hat = np.zeros(self.n_bins, dtype=np.float64)
        self.beta = np.full(self.n_bins, self.beta_base, dtype=np.float64)
        self.updates = 0
        self.last_mean_e = 0.0
        self.last_n_obs = 0

    def beta_for_z(self, z: torch.Tensor) -> torch.Tensor:
        psi = torch.atan2(z[:, 1], z[:, 0]) % (2 * math.pi)
        psi_np = psi.detach().cpu().numpy()
        vals = periodic_linear_interp(psi_np, self.psi_grid, self.beta)
        out = torch.as_tensor(vals, device=z.device, dtype=z.dtype)
        return out

    def update_from_endpoints(self, z: torch.Tensor, u: torch.Tensor) -> dict:
        """Update from finished skill displacements. Invalid (near-zero) u are skipped."""
        with torch.no_grad():
            nu = u.norm(dim=-1)
            keep = nu > 1e-4
            self.last_n_obs = int(keep.sum().item())
            if self.last_n_obs == 0:
                return {
                    "updated": False,
                    "n_obs": 0,
                    "mean_e": self.last_mean_e,
                    "mean_beta": float(self.beta.mean()),
                    "max_beta": float(self.beta.max()),
                    "min_beta": float(self.beta.min()),
                    "eta": self.eta,
                }
            z_k = z[keep]
            u_k = u[keep]
            e = angular_error_from_u(z_k, u_k).detach().cpu().numpy()
            psi = torch.atan2(z_k[:, 1], z_k[:, 0]).detach().cpu().numpy() % (2 * math.pi)
            e_hat = periodic_linear_interp(self.psi_grid, psi, e)
            e_bar = float(e_hat.mean())
            self.last_mean_e = e_bar
            w = np.power(1.0 + e_hat, self.eta)
            denom = float(w.mean())
            if denom < 1e-12:
                beta = np.full_like(e_hat, self.beta_base)
            else:
                beta = self.beta_base * w / denom
            lo = 0.5 * self.beta_base
            hi = 2.0 * self.beta_base
            self.e_hat = e_hat
            # Formula preserves the circular mean before clipping.
            self.beta = np.clip(beta, lo, hi)
            self.updates += 1
            return {
                "updated": True,
                "n_obs": self.last_n_obs,
                "mean_e": e_bar,
                "mean_beta": float(self.beta.mean()),
                "max_beta": float(self.beta.max()),
                "min_beta": float(self.beta.min()),
                "eta": self.eta,
            }

    def snapshot(self) -> dict:
        return {
            "beta_base": self.beta_base,
            "eta": self.eta,
            "updates": self.updates,
            "mean_e": self.last_mean_e,
            "psi": self.psi_grid.copy(),
            "e_hat": self.e_hat.copy(),
            "beta": self.beta.copy(),
        }


class AdaptiveConeAlpha:
    """Cone half-angle integral control (dual ascent) on angular bins.

    Parameters
    ----------
    beta_base : float
        Warm-start via alpha0 = arctan(beta_base).
    theta_tol : float
        Target cone-ratio angle in radians.
    kappa : float
        Dual step size (≈0.5 recommended under policy lag).
    alpha_max : float
        Upper clip in radians (default 70°).
    only_increase : bool
        If True, keep alpha0 as a floor (conservative for moving-start).
    """

    def __init__(
        self,
        beta_base: float,
        theta_tol: float,
        n_bins: int = 72,
        device=None,
        kappa: float = 0.5,
        alpha_max: float = math.radians(70.0),
        n_min: int = 10,
        only_increase: bool = False,
    ):
        self.beta_base = float(beta_base)
        self.theta_tol = float(theta_tol)
        self.n_bins = int(n_bins)
        self.device = device
        self.kappa = float(kappa)
        self.alpha_max = float(alpha_max)
        self.n_min = int(n_min)
        self.only_increase = bool(only_increase)

        alpha0 = float(np.arctan(self.beta_base))
        alpha0 = float(np.clip(alpha0, 0.0, self.alpha_max))
        self.alpha0 = alpha0
        self.alpha_floor = alpha0 if self.only_increase else 0.0

        self.psi_grid = np.linspace(0.0, 2 * math.pi, self.n_bins, endpoint=False)
        self.alpha = np.full(self.n_bins, alpha0, dtype=np.float64)
        self.beta = np.tan(self.alpha)
        self.hat_theta = np.zeros(self.n_bins, dtype=np.float64)
        self.bin_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.updates = 0
        self.last_mean_e = 0.0
        self.last_n_obs = 0
        self.last_mean_theta = 0.0

    def beta_for_z(self, z: torch.Tensor) -> torch.Tensor:
        psi = torch.atan2(z[:, 1], z[:, 0]) % (2 * math.pi)
        psi_np = psi.detach().cpu().numpy()
        # Interpolate alpha then map to beta (keeps cone geometry smooth).
        alpha_q = periodic_linear_interp(psi_np, self.psi_grid, self.alpha)
        beta_q = np.tan(np.clip(alpha_q, 0.0, self.alpha_max - 1e-6))
        return torch.as_tensor(beta_q, device=z.device, dtype=z.dtype)

    def alpha_for_z(self, z: torch.Tensor) -> torch.Tensor:
        psi = torch.atan2(z[:, 1], z[:, 0]) % (2 * math.pi)
        psi_np = psi.detach().cpu().numpy()
        alpha_q = periodic_linear_interp(psi_np, self.psi_grid, self.alpha)
        return torch.as_tensor(alpha_q, device=z.device, dtype=z.dtype)

    def update_from_endpoints(self, z: torch.Tensor, u: torch.Tensor) -> dict:
        """Integral dual ascent on bins with enough samples."""
        with torch.no_grad():
            nu = u.norm(dim=-1)
            keep = nu > 1e-4
            self.last_n_obs = int(keep.sum().item())
            empty = {
                "updated": False,
                "n_obs": self.last_n_obs,
                "mean_e": self.last_mean_e,
                "mean_theta": self.last_mean_theta,
                "mean_beta": float(self.beta.mean()),
                "max_beta": float(self.beta.max()),
                "min_beta": float(self.beta.min()),
                "mean_alpha_deg": float(np.degrees(self.alpha.mean())),
                "theta_tol_deg": float(np.degrees(self.theta_tol)),
                "n_bins_updated": 0,
            }
            if self.last_n_obs == 0:
                return empty

            z_k = z[keep]
            u_k = u[keep]
            a, b = forward_lateral(z_k, u_k)
            a_np = a.detach().cpu().numpy()
            b_np = b.detach().cpu().numpy()
            psi = torch.atan2(z_k[:, 1], z_k[:, 0]).detach().cpu().numpy() % (2 * math.pi)
            idx = (psi / (2 * math.pi) * self.n_bins).astype(np.int64) % self.n_bins

            n = np.bincount(idx, minlength=self.n_bins).astype(np.int64)
            sum_a = np.bincount(idx, weights=a_np, minlength=self.n_bins).astype(np.float64)
            sum_b = np.bincount(idx, weights=b_np, minlength=self.n_bins).astype(np.float64)
            self.bin_counts = n

            # Cone-ratio angle; undefined / unstable if forward mass is non-positive.
            valid = (n >= self.n_min) & (sum_a > 1e-8)
            th = np.zeros(self.n_bins, dtype=np.float64)
            th[valid] = np.arctan2(sum_b[valid], sum_a[valid])
            self.hat_theta = th

            # Also keep a soft error signal for logging compatibility.
            e = angular_error_from_u(z_k, u_k).detach().cpu().numpy()
            self.last_mean_e = float(e.mean()) if len(e) else 0.0
            self.last_mean_theta = float(th[valid].mean()) if bool(valid.any()) else 0.0

            n_upd = int(valid.sum())
            if n_upd == 0:
                empty["mean_e"] = self.last_mean_e
                empty["mean_theta"] = self.last_mean_theta
                return empty

            alpha_new = self.alpha.copy()
            alpha_new[valid] = self.alpha[valid] + self.kappa * (th[valid] - self.theta_tol)
            lo = self.alpha_floor
            alpha_new = np.clip(alpha_new, lo, self.alpha_max)
            self.alpha = alpha_new
            self.beta = np.tan(self.alpha)
            self.updates += 1
            return {
                "updated": True,
                "n_obs": self.last_n_obs,
                "mean_e": self.last_mean_e,
                "mean_theta": self.last_mean_theta,
                "mean_beta": float(self.beta.mean()),
                "max_beta": float(self.beta.max()),
                "min_beta": float(self.beta.min()),
                "mean_alpha_deg": float(np.degrees(self.alpha.mean())),
                "max_alpha_deg": float(np.degrees(self.alpha.max())),
                "min_alpha_deg": float(np.degrees(self.alpha.min())),
                "theta_tol_deg": float(np.degrees(self.theta_tol)),
                "n_bins_updated": n_upd,
            }

    def snapshot(self) -> dict:
        return {
            "kind": "cone",
            "beta_base": self.beta_base,
            "theta_tol": self.theta_tol,
            "theta_tol_deg": float(np.degrees(self.theta_tol)),
            "kappa": self.kappa,
            "alpha_max": self.alpha_max,
            "alpha0": self.alpha0,
            "only_increase": self.only_increase,
            "updates": self.updates,
            "mean_e": self.last_mean_e,
            "mean_theta": self.last_mean_theta,
            "psi": self.psi_grid.copy(),
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "hat_theta": self.hat_theta.copy(),
            "bin_counts": self.bin_counts.copy(),
        }


def estimate_theta_tol_from_endpoints(
    z: torch.Tensor,
    u: torch.Tensor,
    n_bins: int = 72,
    n_min: int = 10,
    quantile: float = 0.5,
) -> float:
    """Median (or other quantile) of per-bin cone-ratio angles; for setting theta_tol."""
    with torch.no_grad():
        keep = u.norm(dim=-1) > 1e-4
        if int(keep.sum().item()) == 0:
            return math.radians(5.0)
        z_k, u_k = z[keep], u[keep]
        a, b = forward_lateral(z_k, u_k)
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        psi = torch.atan2(z_k[:, 1], z_k[:, 0]).detach().cpu().numpy() % (2 * math.pi)
        idx = (psi / (2 * math.pi) * n_bins).astype(np.int64) % n_bins
        n = np.bincount(idx, minlength=n_bins)
        sum_a = np.bincount(idx, weights=a_np, minlength=n_bins)
        sum_b = np.bincount(idx, weights=b_np, minlength=n_bins)
        valid = (n >= n_min) & (sum_a > 1e-8)
        if not bool(valid.any()):
            return float(np.arctan2(float(b_np.sum()), max(float(a_np.sum()), 1e-8)))
        th = np.arctan2(sum_b[valid], sum_a[valid])
        return float(np.quantile(th, quantile))
