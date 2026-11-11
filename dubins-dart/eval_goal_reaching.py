# -*- coding: utf-8 -*-
"""Zero-shot goal reaching on Dubins car: compare replanning rules."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CKPT = RESULTS / "checkpoints"

from dubins_car.dynamics import DT, OBS_DIM, V, horizon_steps
from dubins_car.torch_dyn import make_obs_torch, step_dynamics_torch
from dubins_car.train_sac import Actor


POLICIES = {
    10: [
        ("METRA", "H10_metra.pt"),
        ("DART const b=0.28", "H10_dart_b0.28.pt"),
        ("DART const b=0.36", "H10_dart_b0.36.pt"),
        ("DART const b=0.64", "H10_dart_b0.64.pt"),
        ("Squared l=0.2", "H10_squared_0.2.pt"),
        ("Norm l=1", "H10_norm_1.pt"),
        ("DART adaptive b0=0.50", "H10_adaptive_b0.50.pt"),
    ],
    6: [
        ("METRA", "H6_metra.pt"),
        ("DART const b=0.74", "H6_dart_b0.74.pt"),
        ("DART const b=1.3", "H6_dart_b1.3.pt"),
        ("DART const b=1.65", "H6_dart_b1.65.pt"),
        ("Squared l=1", "H6_squared_1.pt"),
        ("Norm l=5", "H6_norm_5.pt"),
        ("DART adaptive b0=1.3", "H6_adaptive_b1.3.pt"),
    ],
}

# Controllers to compare.
# fixed: replan only every segment_h steps
# vanilla: paper projection trigger a >= R
# gated (A): a >= R and lateral b <= b_max and j >= j_min
# overshoot (B): a >= R + delta and j >= j_min
# near (C): gated in cruise; when dist < d_near, replan every k steps
# stall (D): j >= j_min and true distance failed to decrease enough
# gated_near (A+C): gated trigger + near-goal dense replanning
CONTROLLERS = ["fixed", "vanilla", "gated", "overshoot", "near", "stall", "gated_near"]


def load_actor(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(OBS_DIM, 256).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def sample_goals(n: int, path_length: float, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0.0, 2 * math.pi, size=n)
    rad = rng.uniform(0.25 * path_length, 0.85 * path_length, size=n)
    g = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=-1).astype(np.float32)
    return torch.from_numpy(g)


@torch.no_grad()
def run_controller(
    actor,
    goals: torch.Tensor,
    train_horizon: int,
    path_length: float,
    segment_h: int,
    max_steps: int,
    success_tol: float,
    mode: str,
    device,
    b_max: float = 0.75,
    overshoot_delta: float = 0.5,
    j_min: int | None = None,
    d_near: float = 1.5,
    near_every: int = 5,
    stall_frac: float = 0.05,
):
    if j_min is None:
        j_min = max(10, segment_h // 3)

    n = goals.shape[0]
    state = torch.zeros(n, 3, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    reached_any = torch.zeros(n, dtype=torch.bool, device=device)
    time_to_reach = torch.full((n,), float(max_steps + 1), device=device)
    replans = torch.zeros(n, dtype=torch.long, device=device)

    z = torch.zeros(n, 2, device=device)
    R = torch.zeros(n, device=device)
    seg_start_xy = torch.zeros(n, 2, device=device)
    seg_start_dist = torch.zeros(n, device=device)
    seg_t = torch.zeros(n, dtype=torch.long, device=device)
    need_plan = torch.ones(n, dtype=torch.bool, device=device)
    steps_since_replan = torch.zeros(n, dtype=torch.long, device=device)

    for step in range(max_steps):
        xy = state[:, :2]
        dist = (goals - xy).norm(dim=-1)
        newly = active & (dist <= success_tol) & (~reached_any)
        if newly.any():
            reached_any = reached_any | newly
            time_to_reach = torch.where(newly, torch.full_like(time_to_reach, float(step)), time_to_reach)

        if need_plan.any():
            idx = need_plan
            delta = goals[idx] - state[idx, :2]
            r = delta.norm(dim=-1).clamp_min(1e-8)
            z[idx] = delta / r.unsqueeze(-1)
            R[idx] = r
            seg_start_xy[idx] = state[idx, :2]
            seg_start_dist[idx] = r
            seg_t[idx] = 0
            steps_since_replan[idx] = 0
            replans[idx] = replans[idx] + 1
            need_plan[idx] = False

        obs = make_obs_torch(
            state,
            z,
            state[:, :2] - seg_start_xy,
            seg_t,
            train_horizon,
            path_length,
        )
        action = actor.deterministic(obs).squeeze(-1)
        next_state = step_dynamics_torch(state, action)
        state = torch.where(active.unsqueeze(-1), next_state, state)
        seg_t = torch.where(active, seg_t + 1, seg_t)
        steps_since_replan = torch.where(active, steps_since_replan + 1, steps_since_replan)

        xy = state[:, :2]
        dist = (goals - xy).norm(dim=-1)
        newly = active & (dist <= success_tol) & (~reached_any)
        if newly.any():
            reached_any = reached_any | newly
            time_to_reach = torch.where(newly, torch.full_like(time_to_reach, float(step + 1)), time_to_reach)

        disp = xy - seg_start_xy
        a_prog = (z * disp).sum(-1)
        # Lateral absolute deviation from the commanded ray.
        b_lat = (-z[:, 1] * disp[:, 0] + z[:, 0] * disp[:, 1]).abs()
        long_enough = seg_t >= j_min
        timed_out = active & (seg_t >= segment_h)

        if mode == "fixed":
            hit = torch.zeros_like(active)
        elif mode == "vanilla":
            hit = active & (a_prog >= R)
        elif mode == "gated":
            hit = active & long_enough & (a_prog >= R) & (b_lat <= b_max)
        elif mode == "overshoot":
            hit = active & long_enough & (a_prog >= (R + overshoot_delta))
        elif mode == "near":
            # Pure near-goal densification: no mid-range projection trigger.
            hit = active & (dist < d_near) & (steps_since_replan >= near_every)
        elif mode == "stall":
            expected_drop = stall_frac * seg_t.float() * (V * DT)  # loose lower bound on useful progress
            no_progress = dist > (seg_start_dist - expected_drop).clamp_min(0.0)
            hit = active & long_enough & no_progress
        elif mode == "gated_near":
            # A + C: lateral-gated projection in cruise, dense replan near the goal.
            cruise = active & long_enough & (a_prog >= R) & (b_lat <= b_max) & (dist >= d_near)
            dense = active & (dist < d_near) & (steps_since_replan >= near_every)
            hit = cruise | dense
        else:
            raise ValueError(mode)

        need_plan = (need_plan | hit | timed_out) & active
        inside = dist <= success_tol
        active = active & (~inside)
        if not bool(active.any()):
            break

    final_xy = state[:, :2]
    final_dist = (goals - final_xy).norm(dim=-1)
    final_success = final_dist <= success_tol
    high_level = (replans - 1).clamp_min(0)
    ttr = torch.where(reached_any, time_to_reach, torch.full_like(time_to_reach, float("nan")))
    return {
        "final_success": float(final_success.float().mean()),
        "any_success": float(reached_any.float().mean()),
        "final_dist_mean": float(final_dist.mean()),
        "final_dist_med": float(final_dist.median()),
        "time_to_reach_mean": float(ttr.nanmean()) if bool(reached_any.any()) else float("nan"),
        "replans_mean": float(high_level.float().mean()),
    }


def score_tuple(row):
    # Higher success, lower distance, lower time, fewer replans.
    ttr = row["time_to_reach_mean"]
    ttr = 1e6 if ttr != ttr else ttr
    return (-row["final_success"], row["final_dist_mean"], ttr, row["replans_mean"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    success_tol = 0.5
    n_goals = 256
    rows = []

    for horizon_radii, policies in POLICIES.items():
        train_h = horizon_steps(horizon_radii)
        path_length = train_h * DT * V
        segment_h = max(20, train_h // 5)
        max_steps = int(6 * train_h)
        goals = sample_goals(n_goals, path_length, seed=0).to(device)
        print(
            f"\nH={horizon_radii} train_h={train_h} segment_h={segment_h} max_steps={max_steps}",
            flush=True,
        )
        for name, fname in policies:
            path = CKPT / fname
            if not path.exists():
                print(f"  missing {fname}", flush=True)
                continue
            actor = load_actor(path, device)
            for mode in CONTROLLERS:
                stats = run_controller(
                    actor,
                    goals,
                    train_h,
                    path_length,
                    segment_h,
                    max_steps,
                    success_tol,
                    mode,
                    device,
                )
                row = {
                    "horizon": horizon_radii,
                    "policy": name,
                    "controller": mode,
                    **stats,
                }
                rows.append(row)
                print(
                    f"  {name:26s} {mode:11s}  "
                    f"final={row['final_success']:.3f}  dist={row['final_dist_mean']:.3f}  "
                    f"ttr={row['time_to_reach_mean']:.1f}  replans={row['replans_mean']:.1f}",
                    flush=True,
                )
            del actor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_csv = RESULTS / "goal_reaching_controllers.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate: for each horizon x controller, average over policies.
    lines = [
        "# Controller comparison on Dubins zero-shot goal reaching",
        "",
        f"Goals={n_goals}, tol={success_tol}, seed=0.",
        "",
        "Controllers:",
        "- fixed: replan only at segment timeout",
        "- vanilla: paper projection trigger (a >= R)",
        "- gated (A): a >= R and lateral b <= 0.75 and j >= j_min",
        "- overshoot (B): a >= R+0.5 and j >= j_min",
        "- near (C): gated cruise + dense replan when dist < 1.5",
        "- stall (D): replan if distance does not drop enough after j_min",
        "- gated_near (A+C): gated + near-goal dense replan",
        "",
    ]

    # Per-horizon mean over policies
    lines.append("## Average over all policies")
    lines.append("")
    for h in (10, 6):
        lines.append(f"### H={h} r_min")
        lines.append("")
        lines.append("| Controller | Final succ. | Final dist. | Time to reach | Replans |")
        lines.append("|---|---:|---:|---:|---:|")
        subset = [r for r in rows if r["horizon"] == h]
        ranked = []
        for mode in CONTROLLERS:
            ms = [r for r in subset if r["controller"] == mode]
            agg = {
                "controller": mode,
                "final_success": float(np.mean([r["final_success"] for r in ms])),
                "final_dist_mean": float(np.mean([r["final_dist_mean"] for r in ms])),
                "time_to_reach_mean": float(np.nanmean([r["time_to_reach_mean"] for r in ms])),
                "replans_mean": float(np.mean([r["replans_mean"] for r in ms])),
            }
            ranked.append(agg)
        ranked.sort(key=score_tuple)
        for agg in ranked:
            lines.append(
                f"| {agg['controller']} | {agg['final_success']:.3f} | {agg['final_dist_mean']:.3f} | "
                f"{agg['time_to_reach_mean']:.1f} | {agg['replans_mean']:.1f} |"
            )
        lines.append("")
        lines.append(f"Best average on H={h}: **{ranked[0]['controller']}**")
        lines.append("")

    # Focus on best DART policies
    focus = {
        10: ["DART const b=0.36", "DART adaptive b0=0.50", "METRA"],
        6: ["DART const b=1.3", "DART adaptive b0=1.3", "METRA"],
    }
    lines.append("## Focus policies")
    lines.append("")
    for h, names in focus.items():
        lines.append(f"### H={h}")
        lines.append("")
        lines.append("| Policy | Controller | Final succ. | Final dist. | Time | Replans |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for name in names:
            cands = [r for r in rows if r["horizon"] == h and r["policy"] == name]
            cands.sort(key=score_tuple)
            for r in cands:
                lines.append(
                    f"| {r['policy']} | {r['controller']} | {r['final_success']:.3f} | "
                    f"{r['final_dist_mean']:.3f} | {r['time_to_reach_mean']:.1f} | {r['replans_mean']:.1f} |"
                )
            best = cands[0]
            lines.append(f"| | **best -> {best['controller']}** | | | | |")
        lines.append("")

    # Overall winner counts: how often each controller is best per policy
    lines.append("## How often each controller wins (per policy)")
    lines.append("")
    win = {m: 0 for m in CONTROLLERS}
    for h in (10, 6):
        pols = sorted({r["policy"] for r in rows if r["horizon"] == h})
        for p in pols:
            cands = [r for r in rows if r["horizon"] == h and r["policy"] == p]
            best = min(cands, key=score_tuple)
            win[best["controller"]] += 1
    lines.append("| Controller | #policy wins |")
    lines.append("|---|---:|")
    for m, c in sorted(win.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {m} | {c} |")
    lines.append("")

    overall_best = max(win.items(), key=lambda kv: kv[1])[0]
    lines.append(f"**Overall most frequent winner: `{overall_best}`**")
    lines.append("")

    md = RESULTS / "goal_reaching_controllers.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "goal_reaching_controllers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Bar chart: average final success by controller
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, h in zip(axes, (10, 6)):
        means = []
        dists = []
        for mode in CONTROLLERS:
            ms = [r for r in rows if r["horizon"] == h and r["controller"] == mode]
            means.append(float(np.mean([r["final_success"] for r in ms])))
            dists.append(float(np.mean([r["final_dist_mean"] for r in ms])))
        x = np.arange(len(CONTROLLERS))
        ax.bar(x - 0.18, means, width=0.36, label="final success")
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, dists, width=0.36, color="C1", alpha=0.7, label="final dist")
        ax.set_xticks(x, CONTROLLERS, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.7, 1.02)
        ax.set_ylabel("final success")
        ax2.set_ylabel("final distance")
        ax.set_title(f"H={h} r_min (avg over policies)")
    fig.suptitle("Replanning-rule comparison")
    fig.savefig(RESULTS / "figures" / "fig_controllers.png", dpi=160)
    plt.close(fig)

    print(f"\nwrote {out_csv}")
    print(f"wrote {md}")
    print(f"wrote {RESULTS / 'figures' / 'fig_controllers.png'}")
    print(f"most wins: {overall_best}")


if __name__ == "__main__":
    main()
