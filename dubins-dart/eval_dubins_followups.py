# -*- coding: utf-8 -*-
"""Dubins follow-up experiments for paper rebuttal / Q3 clarity.

1) Controller ablation: Fixed(h=25) / Distance-Adaptive × {stop, no-stop}
2) Random-start skill robustness (mean angular error + blind gap)
3) β-sweep metrics from frozen constant-DART / METRA checkpoints
4) Replanning counts N_replan under matched budgets
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from dubins_car.dynamics import DT, V, horizon_steps
from dubins_car.metrics import angular_gap, direction_errors
from dubins_car.torch_dyn import make_obs_torch, step_dynamics_torch
from dubins_car.train_sac import evaluate, rollout_deterministic
from eval_adr import adr_horizon, load_actor
from eval_goal_reaching import sample_goals

POLICIES = {
    10: [
        ("DART adaptive", "H10_adaptive_b0.50.pt", "main"),
        ("DART constant", "H10_dart_b0.64.pt", "main"),
        ("METRA", "H10_metra.pt", "main"),
    ],
    6: [
        ("DART adaptive", "H6_adaptive_b1.3.pt", "main"),
        ("DART constant", "H6_dart_b1.3.pt", "main"),
        ("METRA", "H6_metra.pt", "main"),
    ],
}

BETA_SWEEP = {
    10: [
        ("METRA", "H10_metra.pt", 0.0),
        ("DART", "H10_dart_b0.16.pt", 0.16),
        ("DART", "H10_dart_b0.25.pt", 0.25),
        ("DART", "H10_dart_b0.28.pt", 0.28),
        ("DART", "H10_dart_b0.36.pt", 0.36),
        ("DART", "H10_dart_b0.5.pt", 0.5),
        ("DART", "H10_dart_b0.64.pt", 0.64),
        ("DART", "H10_dart_b0.75.pt", 0.75),
        ("DART", "H10_dart_b1.pt", 1.0),
        ("DART", "H10_dart_b1.5.pt", 1.5),
        ("DART", "H10_dart_b2.pt", 2.0),
    ],
    6: [
        ("METRA", "H6_metra.pt", 0.0),
        ("DART", "H6_dart_b0.25.pt", 0.25),
        ("DART", "H6_dart_b0.5.pt", 0.5),
        ("DART", "H6_dart_b0.74.pt", 0.74),
        ("DART", "H6_dart_b0.75.pt", 0.75),
        ("DART", "H6_dart_b1.pt", 1.0),
        ("DART", "H6_dart_b1.3.pt", 1.3),
        ("DART", "H6_dart_b1.5.pt", 1.5),
        ("DART", "H6_dart_b1.65.pt", 1.65),
        ("DART", "H6_dart_b2.pt", 2.0),
        ("DART", "H6_dart_b2.96.pt", 2.96),
    ],
}


def ckpt_path(seed: int, fname: str, kind: str = "main") -> Path:
    if kind == "d3":
        return RESULTS / f"d3_seed{seed}" / "checkpoints" / fname
    if seed == 0:
        return RESULTS / "checkpoints" / fname
    return RESULTS / f"seed{seed}" / "checkpoints" / fname


@torch.no_grad()
def run_goal_controller(
    actor,
    goals: torch.Tensor,
    train_horizon: int,
    path_length: float,
    h_exec: int,
    max_steps: int,
    success_tol: float,
    controller: str,
    stop_on_goal: bool,
    device,
    eps: float = 1e-6,
    preroll_steps: int = 8,
):
    """controller in {fixed, dar}.

    Ablation semantics (important):
      - Adaptive horizon itself: dar always uses h=floor((D-rg)/(2v)), clipped.
      - stop_on_goal=False: never freeze; inside the ball force h>=1 and keep
        replanning (leave-after-arrive allowed). This isolates horizon adaptation.
      - stop_on_goal=True: freeze when D<=rg (or dar sets h=0), holding the pose.
    """
    n = goals.shape[0]
    state = torch.zeros(n, 3, device=device)
    frozen = torch.zeros(n, dtype=torch.bool, device=device)
    reached_any = torch.zeros(n, dtype=torch.bool, device=device)
    time_to_reach = torch.full((n,), float("nan"), device=device)
    replans = torch.zeros(n, dtype=torch.long, device=device)

    z = torch.zeros(n, 2, device=device)
    seg_start = torch.zeros(n, 2, device=device)
    seg_t = torch.zeros(n, dtype=torch.long, device=device)
    seg_path = torch.zeros(n, device=device)
    h_limit = torch.full((n,), max(1, h_exec), dtype=torch.long, device=device)
    need_plan = torch.ones(n, dtype=torch.bool, device=device)

    v_nom = float(V * DT)
    if controller == "dar":
        v_hat = torch.full((n,), v_nom, device=device)
        delta0 = goals - state[:, :2]
        r0 = delta0.norm(dim=-1).clamp_min(1e-8)
        z0 = delta0 / r0.unsqueeze(-1)
        st0 = state.clone()
        u0 = torch.zeros(n, 2, device=device)
        path0 = torch.zeros(n, device=device)
        for t in range(preroll_steps):
            obs = make_obs_torch(
                st0,
                z0,
                u0,
                torch.full((n,), t, device=device, dtype=torch.long),
                train_horizon,
                path_length,
            )
            a = actor.deterministic(obs).squeeze(-1)
            nxt = step_dynamics_torch(st0, a)
            path0 = path0 + (nxt[:, :2] - st0[:, :2]).norm(dim=-1)
            u0 = nxt[:, :2]
            st0 = nxt
        v_hat = (path0 / float(max(preroll_steps, 1))).clamp_min(0.25 * v_nom)
    else:
        v_hat = torch.full((n,), float("nan"), device=device)

    for step in range(max_steps):
        xy = state[:, :2]
        dist = (goals - xy).norm(dim=-1)

        newly = (~frozen) & (dist <= success_tol) & (~reached_any)
        if newly.any():
            reached_any = reached_any | newly
            time_to_reach = torch.where(
                newly, torch.full_like(time_to_reach, float(step)), time_to_reach
            )
            if stop_on_goal:
                frozen = frozen | newly

        if need_plan.any():
            idx = need_plan & (~frozen)
            if bool(idx.any()):
                finished = idx & (replans > 0) & (seg_t > 0)
                if controller == "dar" and bool(finished.any()):
                    v_hat[finished] = seg_path[finished] / seg_t[finished].float().clamp_min(1.0)

                delta = goals[idx] - state[idx, :2]
                r = delta.norm(dim=-1).clamp_min(1e-8)
                z[idx] = delta / r.unsqueeze(-1)
                seg_start[idx] = state[idx, :2]
                seg_t[idx] = 0
                seg_path[idx] = 0.0

                if controller == "fixed":
                    h_limit[idx] = max(1, int(h_exec))
                elif controller == "dar":
                    if stop_on_goal:
                        h = adr_horizon(r, success_tol, v_hat[idx], h_exec, eps)
                        h_limit[idx] = h
                        frozen[idx] = frozen[idx] | (h == 0)
                    else:
                        # Horizon adaptation only: always execute at least 1 step,
                        # even inside the success ball (can leave after arrive).
                        rem = (r - success_tol).clamp_min(0.0)
                        vh = torch.where(
                            torch.isnan(v_hat[idx]) | (v_hat[idx] <= 0),
                            torch.full_like(r, v_nom),
                            v_hat[idx],
                        )
                        raw = torch.floor(rem / (2.0 * vh.clamp_min(0.0) + eps))
                        h_limit[idx] = raw.clamp(min=1.0, max=float(h_exec)).long()
                else:
                    raise ValueError(controller)

                replans[idx] = replans[idx] + 1
            need_plan[:] = False

        moving = ~frozen
        if not bool(moving.any()):
            break

        obs = make_obs_torch(
            state, z, state[:, :2] - seg_start, seg_t, train_horizon, path_length
        )
        action = actor.deterministic(obs).squeeze(-1)
        next_state = step_dynamics_torch(state, action)
        step_len = (next_state[:, :2] - state[:, :2]).norm(dim=-1)
        state = torch.where(moving.unsqueeze(-1), next_state, state)
        seg_path = torch.where(moving, seg_path + step_len, seg_path)
        seg_t = torch.where(moving, seg_t + 1, seg_t)

        xy = state[:, :2]
        dist = (goals - xy).norm(dim=-1)
        newly = moving & (dist <= success_tol) & (~reached_any)
        if newly.any():
            reached_any = reached_any | newly
            time_to_reach = torch.where(
                newly, torch.full_like(time_to_reach, float(step + 1)), time_to_reach
            )
            if stop_on_goal:
                frozen = frozen | newly

        timed_out = moving & (seg_t >= h_limit.clamp_min(1))
        need_plan = timed_out & (~frozen)

        if stop_on_goal:
            frozen = frozen | (dist <= success_tol)
            if not bool((~frozen).any()):
                break

    final_dist = (goals - state[:, :2]).norm(dim=-1)
    final_success = final_dist <= success_tol
    return {
        "final_success": float(final_success.float().mean()),
        "any_success": float(reached_any.float().mean()),
        "final_dist_mean": float(final_dist.mean()),
        "final_dist_reached_mean": float(final_dist[reached_any].mean())
        if bool(reached_any.any())
        else float("nan"),
        "hold_gap": float((reached_any & ~final_success).float().mean()),
        "n_replan": float(replans.float().mean()),
        "time_to_reach_mean": float(time_to_reach.nanmean())
        if bool(reached_any.any())
        else float("nan"),
    }


@torch.no_grad()
def eval_random_starts(
    actor,
    train_horizon: int,
    path_length: float,
    device,
    n_starts: int = 64,
    k_skills: int = 180,
    seed: int = 0,
    pos_scale: float = 0.5,
):
    """Mean skill angular error / blind gap over random initial poses."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # Random heading + mild position offset (still near origin).
    th0 = torch.rand(n_starts, generator=g, device=device) * (2 * math.pi)
    xy0 = (torch.rand(n_starts, 2, generator=g, device=device) - 0.5) * (2 * pos_scale)
    starts = torch.cat((xy0, th0.unsqueeze(-1)), dim=-1)

    psi = torch.arange(k_skills, device=device, dtype=torch.float32) * (
        2 * math.pi / k_skills
    )
    z_unit = torch.stack((psi.cos(), psi.sin()), dim=-1)  # world-frame skill codes

    errs = []
    blinds = []
    forwards = []
    radii = []
    for i in range(n_starts):
        s0 = starts[i]
        # Rotate skill codes into world frame relative to initial heading:
        # keep z as absolute world directions (same as fixed-start protocol),
        # but start from random pose.
        b = k_skills
        state = s0.unsqueeze(0).repeat(b, 1)
        u0 = state[:, :2].clone()
        u = torch.zeros(b, 2, device=device)
        t = torch.zeros(b, dtype=torch.long, device=device)
        for _ in range(train_horizon):
            obs = make_obs_torch(state, z_unit, u, t, train_horizon, path_length)
            action = actor.deterministic(obs).squeeze(-1)
            state = step_dynamics_torch(state, action)
            u = state[:, :2] - u0
            t = t + 1
        z_np = z_unit.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        err = direction_errors(z_np, u_np)
        gap = angular_gap(u_np)
        errs.append(float(np.degrees(err.mean())))
        blinds.append(float(gap["gap_deg"]))
        forwards.append(float(np.sum(z_np * u_np, axis=1).mean()))
        radii.append(float(np.linalg.norm(u_np, axis=1).mean()))

    return {
        "mean_err_deg": float(np.mean(errs)),
        "mean_err_deg_std": float(np.std(errs, ddof=1)),
        "blind_deg": float(np.mean(blinds)),
        "blind_deg_std": float(np.std(blinds, ddof=1)),
        "mean_forward": float(np.mean(forwards)),
        "endpoint_radius": float(np.mean(radii)),
        "n_starts": n_starts,
        "k_skills": k_skills,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-goals", type=int, default=256)
    ap.add_argument("--n-starts", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--parts",
        default="controller,random,beta,replan",
        help="comma list: controller,random,beta,replan",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    out = RESULTS / "dubins_followups"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"device={device} seeds={seeds} parts={sorted(parts)}", flush=True)

    controller_rows = []
    random_rows = []
    beta_rows = []

    # ---- 1+4 controller ablation + replan ----
    if "controller" in parts or "replan" in parts:
        ablations = [
            ("Fixed (h=25)", "fixed", 25, False),
            ("Fixed (h=25) + stop", "fixed", 25, True),
            ("Fixed (h=1)", "fixed", 1, False),
            ("Fixed (h=1) + stop", "fixed", 1, True),
            ("Distance-Adaptive", "dar", None, False),
            ("Distance-Adaptive + stop", "dar", None, True),
        ]
        for seed in seeds:
            for H, pols in POLICIES.items():
                train_h = horizon_steps(H)
                path_length = train_h * DT * V
                default_h_max = max(20, train_h // 5)
                max_steps = int(6 * train_h)
                goals = sample_goals(args.n_goals, path_length, seed=0).to(device)
                print(f"\n=== controller seed={seed} H={H} ===", flush=True)
                for pname, fname, kind in pols:
                    path = ckpt_path(seed, fname, kind)
                    if not path.exists():
                        print(f"  missing {path}", flush=True)
                        continue
                    actor = load_actor(path, device)
                    for label, ctrl, h_fixed, stop in ablations:
                        h_exec = default_h_max if h_fixed is None else int(h_fixed)
                        st = run_goal_controller(
                            actor,
                            goals,
                            train_h,
                            path_length,
                            h_exec,
                            max_steps,
                            0.5,
                            ctrl,
                            stop,
                            device,
                        )
                        row = {
                            "seed": seed,
                            "horizon": H,
                            "policy": pname,
                            "controller": label,
                            "stop_on_goal": int(stop),
                            "h_exec": h_exec,
                            **st,
                        }
                        controller_rows.append(row)
                        print(
                            f"  {pname:14s} {label:28s} "
                            f"final={st['final_success']:.3f} any={st['any_success']:.3f} "
                            f"dist={st['final_dist_mean']:.3f} N={st['n_replan']:.1f} "
                            f"hold={st['hold_gap']:.3f}",
                            flush=True,
                        )
                    del actor

        with (out / "controller_ablation.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(controller_rows[0].keys()))
            w.writeheader()
            for r in controller_rows:
                w.writerow(r)

    # ---- 2 random-start ----
    if "random" in parts:
        for seed in seeds:
            for H, pols in POLICIES.items():
                train_h = horizon_steps(H)
                path_length = train_h * DT * V
                print(f"\n=== random-start seed={seed} H={H} ===", flush=True)
                for pname, fname, kind in pols:
                    path = ckpt_path(seed, fname, kind)
                    if not path.exists():
                        continue
                    actor = load_actor(path, device)
                    # Fixed-start baseline
                    ev0 = evaluate(actor, train_h, path_length, device, k=360)
                    rad0 = float(np.linalg.norm(ev0["u"], axis=1).mean())
                    rnd = eval_random_starts(
                        actor,
                        train_h,
                        path_length,
                        device,
                        n_starts=args.n_starts,
                        k_skills=180,
                        seed=seed,
                    )
                    row = {
                        "seed": seed,
                        "horizon": H,
                        "policy": pname,
                        "fixed_mean_err": ev0["mean_err_deg"],
                        "fixed_blind": ev0["blind_deg"],
                        "fixed_forward": ev0["mean_forward"],
                        "fixed_radius": rad0,
                        **{f"random_{k}": v for k, v in rnd.items()},
                    }
                    random_rows.append(row)
                    print(
                        f"  {pname:14s} fixed_err={ev0['mean_err_deg']:.2f} "
                        f"rand_err={rnd['mean_err_deg']:.2f}±{rnd['mean_err_deg_std']:.2f}  "
                        f"fixed_blind={ev0['blind_deg']:.1f} "
                        f"rand_blind={rnd['blind_deg']:.1f}±{rnd['blind_deg_std']:.1f}",
                        flush=True,
                    )
                    del actor

        with (out / "random_start.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(random_rows[0].keys()))
            w.writeheader()
            for r in random_rows:
                w.writerow(r)

    # ---- 3 beta sweep from existing ckpts ----
    if "beta" in parts:
        for seed in seeds:
            for H, items in BETA_SWEEP.items():
                train_h = horizon_steps(H)
                path_length = train_h * DT * V
                print(f"\n=== beta sweep seed={seed} H={H} ===", flush=True)
                for mode_name, fname, beta in items:
                    path = ckpt_path(seed, fname)
                    if not path.exists():
                        print(f"  missing {path}", flush=True)
                        continue
                    actor = load_actor(path, device)
                    ev = evaluate(actor, train_h, path_length, device, k=360)
                    rad = float(np.linalg.norm(ev["u"], axis=1).mean())
                    row = {
                        "seed": seed,
                        "horizon": H,
                        "mode": mode_name,
                        "beta": beta,
                        "mean_err_deg": ev["mean_err_deg"],
                        "max_err_deg": ev["max_err_deg"],
                        "blind_deg": ev["blind_deg"],
                        "mean_forward": ev["mean_forward"],
                        "endpoint_radius": rad,
                        "bound_deg": ev.get("bound_deg", float("nan")),
                    }
                    beta_rows.append(row)
                    print(
                        f"  beta={beta:<5g} mean={ev['mean_err_deg']:.2f} "
                        f"blind={ev['blind_deg']:.1f} rad={rad:.3f} fwd={ev['mean_forward']:.3f}",
                        flush=True,
                    )
                    del actor

        with (out / "beta_sweep.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(beta_rows[0].keys()))
            w.writeheader()
            for r in beta_rows:
                w.writerow(r)

    # ---- markdown summary ----
    lines = [
        "# Dubins follow-up experiments",
        "",
        f"Wall={(time.perf_counter()-t0)/60:.1f} min. device={device}.",
        "",
    ]
    if controller_rows:
        lines += [
            "## Controller ablation (mean over seeds)",
            "",
            "| H | Policy | Controller | final | any | final_dist | N_replan | hold_gap |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        keys = sorted(
            {
                (r["horizon"], r["policy"], r["controller"])
                for r in controller_rows
            }
        )
        for H, pol, ctrl in keys:
            xs = [
                r
                for r in controller_rows
                if r["horizon"] == H and r["policy"] == pol and r["controller"] == ctrl
            ]
            def m(k):
                return float(np.mean([r[k] for r in xs]))

            lines.append(
                f"| {H} | {pol} | {ctrl} | {m('final_success'):.3f} | {m('any_success'):.3f} | "
                f"{m('final_dist_mean'):.3f} | {m('n_replan'):.1f} | {m('hold_gap'):.3f} |"
            )
        lines.append("")

    if random_rows:
        lines += [
            "## Random-start robustness (mean over seeds)",
            "",
            "| H | Policy | fixed_err | rand_err | fixed_blind | rand_blind |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        keys = sorted({(r["horizon"], r["policy"]) for r in random_rows})
        for H, pol in keys:
            xs = [r for r in random_rows if r["horizon"] == H and r["policy"] == pol]
            lines.append(
                f"| {H} | {pol} | {np.mean([r['fixed_mean_err'] for r in xs]):.2f} | "
                f"{np.mean([r['random_mean_err_deg'] for r in xs]):.2f} | "
                f"{np.mean([r['fixed_blind'] for r in xs]):.1f} | "
                f"{np.mean([r['random_blind_deg'] for r in xs]):.1f} |"
            )
        lines.append("")

    if beta_rows:
        lines += [
            "## β sweep (mean over seeds)",
            "",
            "| H | beta | mean_err | blind | radius | forward |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        keys = sorted({(r["horizon"], r["beta"]) for r in beta_rows})
        for H, beta in keys:
            xs = [r for r in beta_rows if r["horizon"] == H and r["beta"] == beta]
            lines.append(
                f"| {H} | {beta:g} | {np.mean([r['mean_err_deg'] for r in xs]):.2f} | "
                f"{np.mean([r['blind_deg'] for r in xs]):.1f} | "
                f"{np.mean([r['endpoint_radius'] for r in xs]):.3f} | "
                f"{np.mean([r['mean_forward'] for r in xs]):.3f} |"
            )

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
