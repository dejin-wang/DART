# -*- coding: utf-8 -*-
"""ADR = enhanced Distance-Adaptive Replanning (paper targeting controller).

h_k = max{0, min{h_max, floor((D_k - r_g)_+ / (2 v_hat + eps))}}
with the operational rule:
  - if D_k <= r_g: h=0 and freeze (stop inside the success ball)
  - if D_k >  r_g: h = max{1, formula}  (never freeze outside the ball)

Enhancements vs naive DAR:
  1) Pre-roll / nominal init of v_hat (no blind first h_max)
  2) h=0 freeze when inside success ball (prevent leave-after-arrive)
  3) MultiGoals: keep v_hat across goals
"""

from __future__ import annotations

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

from dubins_car.dynamics import DT, OBS_DIM, V, horizon_steps
from dubins_car.torch_dyn import make_obs_torch, step_dynamics_torch
from dubins_car.train_sac import Actor
from eval_goal_reaching import sample_goals

SUITE = {
    10: [
        ("METRA", "H10_metra.pt"),
        ("DART const b=0.36", "H10_dart_b0.36.pt"),
        ("DART adaptive b0=0.50", "H10_adaptive_b0.50.pt"),
    ],
    6: [
        ("METRA", "H6_metra.pt"),
        ("DART const b=1.3", "H6_dart_b1.3.pt"),
        ("DART adaptive b0=1.3", "H6_adaptive_b1.3.pt"),
    ],
}


def ckpt_dir(seed: int) -> Path:
    return RESULTS / "checkpoints" if seed == 0 else RESULTS / f"seed{seed}" / "checkpoints"


def load_actor(path: Path, device):
    import numpy.core as _nc

    sys.modules.setdefault("numpy._core", _nc)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    try:
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
    except Exception:
        pass
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(OBS_DIM, 256).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def adr_horizon(r: torch.Tensor, success_tol: float, v_hat: torch.Tensor, h_max: int, eps: float) -> torch.Tensor:
    """Enhanced ADR horizon. Never freeze outside the success ball."""
    inside = r <= success_tol
    rem = (r - success_tol).clamp_min(0.0)
    raw = torch.floor(rem / (2.0 * v_hat.clamp_min(0.0) + eps))
    h = raw.clamp(min=1.0, max=float(h_max)).long()
    return torch.where(inside, torch.zeros_like(h), h)


@torch.no_grad()
def run_single(
    actor,
    goals: torch.Tensor,
    train_horizon: int,
    path_length: float,
    h_max: int,
    max_steps: int,
    success_tol: float,
    mode: str,
    device,
    h_near: int = 5,
    eps: float = 1e-6,
    preroll_steps: int = 8,
):
    """mode: fixed | ahr | dar | adr"""
    n = goals.shape[0]
    state = torch.zeros(n, 3, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    reached_any = torch.zeros(n, dtype=torch.bool, device=device)
    time_to_reach = torch.full((n,), float(max_steps + 1), device=device)
    replans = torch.zeros(n, dtype=torch.long, device=device)

    z = torch.zeros(n, 2, device=device)
    seg_start_xy = torch.zeros(n, 2, device=device)
    seg_t = torch.zeros(n, dtype=torch.long, device=device)
    h_limit = torch.full((n,), h_max, dtype=torch.long, device=device)
    need_plan = torch.ones(n, dtype=torch.bool, device=device)
    seg_path = torch.zeros(n, device=device)
    frozen = torch.zeros(n, dtype=torch.bool, device=device)

    v_nom = float(V * DT)
    if mode == "adr":
        # (1) Pre-roll init v_hat — observation only, does not move episode state.
        v_hat = torch.full((n,), v_nom, device=device)
        delta0 = goals - state[:, :2]
        r0 = delta0.norm(dim=-1).clamp_min(1e-8)
        z0 = delta0 / r0.unsqueeze(-1)
        st0 = state.clone()
        u0 = torch.zeros(n, 2, device=device)
        path0 = torch.zeros(n, device=device)
        for t in range(preroll_steps):
            obs = make_obs_torch(
                st0, z0, u0, torch.full((n,), t, device=device, dtype=torch.long), train_horizon, path_length
            )
            a = actor.deterministic(obs).squeeze(-1)
            nxt = step_dynamics_torch(st0, a)
            path0 = path0 + (nxt[:, :2] - st0[:, :2]).norm(dim=-1)
            u0 = nxt[:, :2]
            st0 = nxt
        v_hat = (path0 / float(max(preroll_steps, 1))).clamp_min(0.25 * v_nom)
    elif mode == "dar":
        v_hat = torch.full((n,), float("nan"), device=device)
    else:
        v_hat = torch.full((n,), float("nan"), device=device)

    switch_dist = 3.0 * success_tol

    for step in range(max_steps):
        xy = state[:, :2]
        dist = (goals - xy).norm(dim=-1)

        newly = active & (~frozen) & (dist <= success_tol) & (~reached_any)
        if newly.any():
            reached_any = reached_any | newly
            time_to_reach = torch.where(newly, torch.full_like(time_to_reach, float(step)), time_to_reach)
            if mode == "adr":
                frozen = frozen | newly
                need_plan = need_plan | newly

        if need_plan.any():
            idx = need_plan
            finished = idx & (replans > 0) & (seg_t > 0) & (~frozen)
            if bool(finished.any()):
                v_hat[finished] = seg_path[finished] / seg_t[finished].float().clamp_min(1.0)

            delta = goals[idx] - state[idx, :2]
            r = delta.norm(dim=-1).clamp_min(1e-8)
            z[idx] = delta / r.unsqueeze(-1)
            seg_start_xy[idx] = state[idx, :2]
            seg_t[idx] = 0
            seg_path[idx] = 0.0

            if mode == "fixed":
                h_limit[idx] = h_max
            elif mode == "ahr":
                far = r > switch_dist
                h_limit[idx] = torch.where(
                    far,
                    torch.full_like(r, h_max, dtype=torch.long),
                    torch.full_like(r, h_near, dtype=torch.long),
                )
            elif mode == "dar":
                rem = (r - success_tol).clamp_min(0.0)
                vh = v_hat[idx]
                use_max = torch.isnan(vh) | (vh <= 0)
                raw = torch.floor(rem / (2.0 * vh.clamp_min(0.0) + eps))
                h = raw.clamp(min=1, max=float(h_max)).long()
                h_limit[idx] = torch.where(use_max, torch.full_like(h, h_max), h)
            elif mode == "adr":
                vh = torch.where(
                    torch.isnan(v_hat[idx]) | (v_hat[idx] <= 0),
                    torch.full_like(r, v_nom),
                    v_hat[idx],
                )
                h = adr_horizon(r, success_tol, vh, h_max, eps)
                h_limit[idx] = h
                frozen[idx] = frozen[idx] | (h == 0)
            else:
                raise ValueError(mode)

            replans[idx] = replans[idx] + 1
            need_plan[idx] = False

        moving = active & (~frozen)
        obs = make_obs_torch(state, z, state[:, :2] - seg_start_xy, seg_t, train_horizon, path_length)
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
            time_to_reach = torch.where(newly, torch.full_like(time_to_reach, float(step + 1)), time_to_reach)
            if mode == "adr":
                frozen = frozen | newly

        timed_out = moving & (h_limit > 0) & (seg_t >= h_limit)
        need_plan = (need_plan | timed_out) & active

        if mode == "adr":
            frozen = frozen | (dist <= success_tol)
            # Episode ends when everyone is frozen (held) or inactive.
            if not bool((active & (~frozen)).any()):
                break
        else:
            active = active & (dist > success_tol)
            if not bool(active.any()):
                break

    final_dist = (goals - state[:, :2]).norm(dim=-1)
    final_success = final_dist <= success_tol
    ttr = torch.where(reached_any, time_to_reach, torch.full_like(time_to_reach, float("nan")))
    return {
        "final_success": float(final_success.float().mean()),
        "any_success": float(reached_any.float().mean()),
        "final_dist_mean": float(final_dist.mean()),
        "time_to_reach_mean": float(ttr.nanmean()) if bool(reached_any.any()) else float("nan"),
        "replans_mean": float((replans - 1).clamp_min(0).float().mean()),
        "hold_gap": float((reached_any & ~final_success).float().mean()),
    }


@torch.no_grad()
def run_multigoal(
    actor,
    train_horizon: int,
    path_length: float,
    h_max: int,
    success_tol: float,
    mode: str,
    device,
    n_episodes: int = 16,
    n_goals: int = 4,
    goal_radius: float = 3.0,
    steps_per_goal: int = 80,
    h_near: int = 5,
    eps: float = 1e-6,
    preroll_steps: int = 8,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    returns = []
    v_nom = float(V * DT)

    for _ep in range(n_episodes):
        state = torch.zeros(1, 3, device=device)
        angs = rng.uniform(0, 2 * math.pi, size=n_goals)
        rads = rng.uniform(0.4 * goal_radius, goal_radius, size=n_goals)
        goals_np = np.stack([rads * np.cos(angs), rads * np.sin(angs)], axis=-1).astype(np.float32)
        score = 0
        v_hat = torch.tensor([v_nom], device=device)
        if mode == "dar":
            v_hat = torch.tensor([float("nan")], device=device)

        for gi in range(n_goals):
            goal = state[:, :2] + torch.as_tensor(goals_np[gi], device=device).view(1, 2)

            if mode == "dar":
                v_hat = torch.tensor([float("nan")], device=device)  # cold start each goal
            # adr: keep v_hat across goals

            if mode == "adr" and gi == 0:
                delta0 = goal - state[:, :2]
                r0 = delta0.norm(dim=-1).clamp_min(1e-8)
                z0 = delta0 / r0.unsqueeze(-1)
                st0 = state.clone()
                u0 = torch.zeros(1, 2, device=device)
                path0 = torch.zeros(1, device=device)
                for t in range(preroll_steps):
                    obs = make_obs_torch(st0, z0, u0, torch.tensor([t], device=device), train_horizon, path_length)
                    a = actor.deterministic(obs).squeeze(-1)
                    nxt = step_dynamics_torch(st0, a)
                    path0 = path0 + (nxt[:, :2] - st0[:, :2]).norm(dim=-1)
                    u0 = nxt[:, :2] - state[:, :2]
                    st0 = nxt
                v_hat = (path0 / float(preroll_steps)).clamp_min(0.25 * v_nom)

            z = torch.zeros(1, 2, device=device)
            seg_start = state[:, :2].clone()
            seg_t = torch.zeros(1, dtype=torch.long, device=device)
            seg_path = torch.zeros(1, device=device)
            h_limit = torch.tensor([h_max], device=device, dtype=torch.long)
            need_plan = torch.tensor([True], device=device)
            frozen = torch.tensor([False], device=device)

            for _step in range(steps_per_goal):
                dist = (goal - state[:, :2]).norm(dim=-1)
                if bool((dist <= success_tol).item()):
                    if mode == "adr":
                        frozen[:] = True

                if bool(need_plan.item()):
                    if int(seg_t.item()) > 0 and not bool(frozen.item()):
                        v_hat = seg_path / seg_t.float().clamp_min(1.0)
                    delta = goal - state[:, :2]
                    r = delta.norm(dim=-1).clamp_min(1e-8)
                    z = delta / r.unsqueeze(-1)
                    seg_start = state[:, :2].clone()
                    seg_t[:] = 0
                    seg_path[:] = 0.0
                    rf = float(r)
                    if mode == "fixed":
                        h_limit[:] = h_max
                    elif mode == "ahr":
                        h_limit[:] = h_max if rf > 3 * success_tol else h_near
                    elif mode == "dar":
                        if torch.isnan(v_hat).item() or float(v_hat) <= 0:
                            h_limit[:] = h_max
                        else:
                            rem = max(0.0, rf - success_tol)
                            h_limit[:] = max(1, min(h_max, int(math.floor(rem / (2.0 * float(v_hat) + eps)))))
                    elif mode == "adr":
                        vh = float(v_hat) if (not torch.isnan(v_hat).item() and float(v_hat) > 0) else v_nom
                        if rf <= success_tol:
                            h = 0
                        else:
                            rem = max(0.0, rf - success_tol)
                            h = max(1, min(h_max, int(math.floor(rem / (2.0 * vh + eps)))))
                        h_limit[:] = h
                        if h == 0:
                            frozen[:] = True
                    need_plan[:] = False

                if bool(frozen.item()):
                    break

                obs = make_obs_torch(state, z, state[:, :2] - seg_start, seg_t, train_horizon, path_length)
                action = actor.deterministic(obs).squeeze(-1)
                nxt = step_dynamics_torch(state, action)
                seg_path = seg_path + (nxt[:, :2] - state[:, :2]).norm(dim=-1)
                state = nxt
                seg_t = seg_t + 1
                if int(h_limit.item()) > 0 and int(seg_t.item()) >= int(h_limit.item()):
                    need_plan[:] = True

            if float((goal - state[:, :2]).norm().item()) <= success_tol:
                score += 1

        returns.append(score)

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "final_success": float(arr.mean() / n_goals),
        "any_success": float((arr > 0).mean()),
        "final_dist_mean": float("nan"),
        "time_to_reach_mean": float("nan"),
        "replans_mean": float("nan"),
        "hold_gap": float("nan"),
        "mg_return": float(arr.mean()),
        "mg_max": float(n_goals),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-goals", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--h-near", type=int, default=5)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    success_tol = 0.5
    modes = [("fixed", "fixed"), ("AHR", "ahr"), ("DAR", "dar"), ("ADR", "adr")]
    rows = []
    t0 = time.perf_counter()
    print(f"device={device} seeds={seeds}  (ADR = enhanced DAR)", flush=True)

    for seed in seeds:
        ckpt = ckpt_dir(seed)
        for Hrad, policies in SUITE.items():
            train_h = horizon_steps(Hrad)
            path_length = train_h * DT * V
            h_max = max(20, train_h // 5)
            max_steps = int(6 * train_h)
            goals = sample_goals(args.n_goals, path_length, seed=0).to(device)
            print(f"\n=== single seed={seed} H={Hrad} h_max={h_max} ===", flush=True)
            for name, fname in policies:
                path = ckpt / fname
                if not path.exists():
                    print(f"  missing {path}", flush=True)
                    continue
                actor = load_actor(path, device)
                for label, mode in modes:
                    st = run_single(
                        actor, goals, train_h, path_length, h_max, max_steps,
                        success_tol, mode, device, h_near=args.h_near,
                    )
                    rows.append(
                        {
                            "protocol": "single",
                            "policy_seed": seed,
                            "horizon": Hrad,
                            "policy": name,
                            "controller": label,
                            **st,
                            "mg_return": float("nan"),
                        }
                    )
                    print(
                        f"  {name:26s} {label:5s}  final={st['final_success']:.3f} any={st['any_success']:.3f} "
                        f"hold_gap={st['hold_gap']:.3f} dist={st['final_dist_mean']:.3f}",
                        flush=True,
                    )
                print("  -- MultiGoals --", flush=True)
                for label, mode in modes:
                    st = run_multigoal(
                        actor, train_h, path_length, h_max, success_tol, mode, device,
                        h_near=args.h_near, seed=seed * 100 + Hrad,
                    )
                    rows.append(
                        {
                            "protocol": "multigoal",
                            "policy_seed": seed,
                            "horizon": Hrad,
                            "policy": name,
                            "controller": label,
                            "final_success": st["final_success"],
                            "any_success": st["any_success"],
                            "final_dist_mean": float("nan"),
                            "time_to_reach_mean": float("nan"),
                            "replans_mean": float("nan"),
                            "hold_gap": float("nan"),
                            "mg_return": st["mg_return"],
                        }
                    )
                    print(
                        f"  {name:26s} {label:5s}  MG={st['mg_return']:.2f}/{st['mg_max']:.0f}",
                        flush=True,
                    )
                del actor
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    out = RESULTS / "adr_comparison"
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    for r in rows:
        for k in keys:
            r.setdefault(k, float("nan"))
    with (out / "adr_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "adr_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def get(proto, seed, H, pol, ctrl):
        for r in rows:
            if (
                r["protocol"] == proto
                and r["policy_seed"] == seed
                and r["horizon"] == H
                and r["policy"] == pol
                and r["controller"] == ctrl
            ):
                return r
        return None

    lines = [
        "# ADR (enhanced DAR) vs DAR / AHR / fixed",
        "",
        f"Wall={(time.perf_counter()-t0)/60:.1f} min. Goals={args.n_goals}, tol={success_tol}.",
        "ADR = pre-roll v_hat + h=0 freeze inside ball + MultiGoals keep v_hat.",
        "",
        "## Single-goal final success",
        "",
        "| seed | H | Policy | fixed | AHR | DAR | ADR | hold_gap ADR |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    d_dar, d_ahr = [], []
    for seed in seeds:
        for H, policies in SUITE.items():
            for name, _ in policies:
                vals = []
                for c in ("fixed", "AHR", "DAR", "ADR"):
                    r = get("single", seed, H, name, c)
                    vals.append(f"{r['final_success']:.3f}" if r else "-")
                r_adr = get("single", seed, H, name, "ADR")
                hg = f"{r_adr['hold_gap']:.3f}" if r_adr else "-"
                lines.append(f"| {seed} | {H} | {name} | " + " | ".join(vals) + f" | {hg} |")
                if r_adr:
                    rd, ra = get("single", seed, H, name, "DAR"), get("single", seed, H, name, "AHR")
                    if rd:
                        d_dar.append(r_adr["final_success"] - rd["final_success"])
                    if ra:
                        d_ahr.append(r_adr["final_success"] - ra["final_success"])

    lines += [
        "",
        "## MultiGoals return (max 4)",
        "",
        "| seed | H | Policy | fixed | AHR | DAR | ADR |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for H, policies in SUITE.items():
            for name, _ in policies:
                vals = []
                for c in ("fixed", "AHR", "DAR", "ADR"):
                    r = get("multigoal", seed, H, name, c)
                    vals.append(f"{r['mg_return']:.2f}" if r and r["mg_return"] == r["mg_return"] else "-")
                lines.append(f"| {seed} | {H} | {name} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Verdict",
        "",
        f"Mean (ADR − DAR) final: **{float(np.mean(d_dar)):+.3f}**",
        f"Mean (ADR − AHR) final: **{float(np.mean(d_ahr)):+.3f}**",
        "",
        "Use **ADR** as the paper targeting controller (enhanced DAR).",
    ]
    md = out / "adr_comparison.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {md}", flush=True)
    print(f"ADR-DAR={float(np.mean(d_dar)):+.3f}  ADR-AHR={float(np.mean(d_ahr)):+.3f}", flush=True)


if __name__ == "__main__":
    main()
