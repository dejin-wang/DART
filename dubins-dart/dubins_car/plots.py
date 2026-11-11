# -*- coding: utf-8 -*-
"""Figures for the Dubins-car direction-alignment experiment."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from dubins_car.metrics import alignment_bound_rad


def _dart_rows(summary):
    rows = [r for r in summary["results"] if r["mode"] == "dart"]
    return sorted(rows, key=lambda r: r["weight"])


def plot_oracle(summary, highlight_weights, ref_L, ref_blind, out_path):
    dart = _dart_rows(summary)
    by_w = {round(r["weight"], 5): r for r in dart}
    chosen = []
    for w in highlight_weights:
        key = round(float(w), 5)
        if key in by_w:
            chosen.append(by_w[key])
    if not chosen:
        chosen = dart[:5]

    n = len(chosen)
    fig = plt.figure(figsize=(3.3 * n, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, n)
    cloud = summary["cloud"]
    L = summary["L"]
    for i, row in enumerate(chosen):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(cloud[:, 0], cloud[:, 1], s=2, c="0.75", alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(row["u"][:, 0], row["u"][:, 1], s=8, c=np.degrees(summary["psi"]), cmap="hsv", linewidths=0)
        if row["blind_deg"] > 8:
            radius = max(1.0, float(np.percentile(np.linalg.norm(row["u"], axis=1), 90)))
            ax.add_patch(
                Wedge(
                    (0, 0),
                    radius,
                    row["blind_start_deg"],
                    row["blind_start_deg"] + row["blind_deg"],
                    facecolor="tab:red",
                    alpha=0.18,
                    edgecolor="none",
                )
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(0, color="0.8", lw=0.4)
        ax.axvline(0, color="0.8", lw=0.4)
        ax.set_title(
            f"beta={row['weight']:g}\nmax err {row['max_err_deg']:.1f} deg, blind {row['blind_deg']:.1f} deg"
        )
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")

    ax = fig.add_subplot(gs[1, 0 : max(1, n // 2)])
    betas = np.array([r["weight"] for r in dart])
    max_err = np.array([r["max_err_deg"] for r in dart])
    mean_err = np.array([r["mean_err_deg"] for r in dart])
    bounds = np.array([np.degrees(alignment_bound_rad(L, b)) for b in betas])
    ref_bounds = np.array([np.degrees(alignment_bound_rad(ref_L, b)) for b in betas])
    ax.plot(betas, max_err, "o-", label="ideal max error")
    ax.plot(betas, mean_err, "s-", label="ideal mean error")
    ax.plot(betas, bounds, "--", label=f"bound using L={L:.2f}")
    ax.plot(betas, ref_bounds, ":", label=f"bound using ref L={ref_L:.2f}")
    ax.axvline(L, color="tab:red", ls="--", lw=0.8, label="measured L")
    ax.set_xlabel("beta")
    ax.set_ylabel("direction error (deg)")
    ax.legend(fontsize=8)
    ax.set_title("Absolute lateral penalty")

    ax = fig.add_subplot(gs[1, max(1, n // 2) :])
    blind = np.array([r["blind_deg"] for r in dart])
    ax.plot(betas, blind, "o-", label="blind sector")
    ax.axhline(ref_blind, color="0.4", ls=":", label=f"reference {ref_blind:.0f} deg")
    ax.axvline(L, color="tab:red", ls="--", lw=0.8, label="measured L")
    ax.set_xlabel("beta")
    ax.set_ylabel("largest endpoint-angle gap (deg)")
    ax.legend(fontsize=8)
    ax.set_title("Blind sector versus beta")
    fig.suptitle(
        f"Dubins car, horizon {summary['horizon_radii']:g} turning radii "
        f"(L_smooth={L:.3f}, L_raw={summary['L_raw']:.3f})"
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_radial(summary, out_path):
    psi_deg = np.degrees(summary["psi"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(psi_deg, summary["rho"])
    axes[0].set_xlabel("direction (deg)")
    axes[0].set_ylabel("rho")
    axes[0].set_title("Outer radial boundary")
    axes[1].plot(psi_deg, summary["dlog_smooth"], label="smoothed")
    axes[1].plot(psi_deg, summary["dlog_raw"], alpha=0.35, label="raw")
    axes[1].axhline(summary["L"], color="tab:red", ls="--", label=f"L={summary['L']:.2f}")
    axes[1].axhline(-summary["L"], color="tab:red", ls="--")
    axes[1].set_xlabel("direction (deg)")
    axes[1].set_ylabel("d log rho / d psi")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Boundary slope")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_other_penalties(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, mode, title in (
        (axes[0], "squared", "Squared lateral penalty"),
        (axes[1], "norm", "Norm matching"),
    ):
        rows = sorted([r for r in summary["results"] if r["mode"] == mode], key=lambda r: r["weight"])
        if not rows:
            continue
        w = [r["weight"] for r in rows]
        ax.plot(w, [r["max_err_deg"] for r in rows], "o-", label="max error")
        ax.plot(w, [r["mean_err_deg"] for r in rows], "s-", label="mean error")
        ax.plot(w, [r["blind_deg"] for r in rows], "^-", label="blind gap")
        ax.set_xlabel("lambda")
        ax.set_ylabel("degrees")
        ax.set_title(title)
        ax.legend(fontsize=8)
    metra = next(r for r in summary["results"] if r["mode"] == "dart" and r["weight"] == 0.0)
    fig.suptitle(
        f"Horizon {summary['horizon_radii']:g}. METRA max error {metra['max_err_deg']:.1f} deg, "
        f"blind {metra['blind_deg']:.1f} deg"
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_error_vs_bound(summary, out_path):
    """Commanded-direction error curves against the uniform worst-case bound."""
    dart = _dart_rows(summary)
    # Show beta = 0, the largest beta still below L, and the smallest beta at or above L, plus the last.
    L = summary["L"]
    picks = [dart[0]]
    below = [r for r in dart if 0 < r["weight"] < L - 1e-9]
    above = [r for r in dart if r["weight"] >= L - 1e-9 and r["weight"] > 0]
    if below:
        picks.append(below[len(below) // 2])
        picks.append(below[-1])
    if above:
        picks.append(above[0])
        if above[-1] is not picks[-1]:
            picks.append(above[-1])
    # Unique by weight, keep order.
    seen = set()
    chosen = []
    for r in picks:
        key = round(r["weight"], 5)
        if key not in seen:
            seen.add(key)
            chosen.append(r)
    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    psi_deg = np.degrees(summary["psi"])
    for ax, row in zip(axes, chosen):
        ax.plot(psi_deg, np.degrees(row["err"]), lw=1.0)
        if row["bound_deg"] == row["bound_deg"]:
            ax.axhline(row["bound_deg"], color="tab:red", ls="--", label=f"bound {row['bound_deg']:.1f} deg")
        ax.set_title(f"beta={row['weight']:g}")
        ax.set_xlabel("commanded angle (deg)")
        ax.set_xlim(0, 360)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("direction error (deg)")
    fig.suptitle(f"Horizon {summary['horizon_radii']:g}: ideal error versus the alignment bound")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training(history, u, psi, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    steps = history["steps"]
    axes[0].plot(steps, history["max_err_deg"], label="max error")
    axes[0].plot(steps, history["mean_err_deg"], label="mean error")
    axes[0].plot(steps, history["blind_deg"], label="blind gap")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("degrees")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Evaluation")
    axes[1].scatter(u[:, 0], u[:, 1], c=np.degrees(psi), cmap="hsv", s=10, linewidths=0)
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].axhline(0, color="0.85", lw=0.4)
    axes[1].axvline(0, color="0.85", lw=0.4)
    axes[1].set_title("Learned endpoints")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_learned_summary(rows, oracle_by_horizon, out_path):
    """rows: list of dicts with horizon, mode, weight, max_err_deg, blind_deg."""
    horizons = sorted({r["horizon"] for r in rows})
    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 3.8 * len(horizons)), constrained_layout=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        subset = [r for r in rows if r["horizon"] == h]
        labels = []
        learned = []
        oracle = []
        bounds = []
        ora = oracle_by_horizon.get(h, {})
        for r in subset:
            labels.append(f"{r['mode']}\n{r['weight']:g}")
            learned.append(r["max_err_deg"])
            match = None
            for item in ora.get("results", []):
                if item["mode"] == ("dart" if r["mode"] == "metra" else r["mode"]) and abs(item["weight"] - r["weight"]) < 1e-6:
                    match = item
                    break
            if r["mode"] == "metra":
                match = next((item for item in ora.get("results", []) if item["mode"] == "dart" and item["weight"] == 0.0), match)
            oracle.append(match["max_err_deg"] if match else math.nan)
            L = ora.get("L", float("nan"))
            if r["mode"] in ("metra", "dart"):
                beta = 0.0 if r["mode"] == "metra" else r["weight"]
                bounds.append(math.degrees(alignment_bound_rad(L, beta)) if L == L else math.nan)
            else:
                bounds.append(math.nan)
        x = np.arange(len(subset))
        ax.bar(x - 0.2, learned, width=0.4, label="SAC max error")
        ax.bar(x + 0.2, oracle, width=0.4, label="ideal max error")
        ax.plot(x, bounds, "r*", label="bound (absolute penalty)")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel("degrees")
        ax.set_title(f"Horizon {h:g} turning radii")
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
