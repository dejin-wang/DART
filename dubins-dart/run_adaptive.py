# -*- coding: utf-8 -*-
"""Train direction-adaptive DART after the constant-beta sweep.

Uses the best constant beta_base values from the finished sweep, then trains
adaptive schedules for both horizons. If a run underperforms its constant twin,
we log a short diagnosis against oracle L and the constant baseline.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from dubins_car.adaptive import AdaptiveBeta, angular_error_from_u  # noqa: E402
from dubins_car.plots import plot_learned_summary, plot_training  # noqa: E402
from dubins_car.train_sac import train_one  # noqa: E402
from run_all import (  # noqa: E402
    json_default,
    load_oracle_scalars,
    load_sac_rows,
    log,
)

# beta_base chosen from the best constant DART runs so far, plus neighbors around L.
ADAPTIVE_JOBS = [
    {"horizon": 10, "weight": 0.28, "name": "H10_adaptive_b0.28", "steps": 150_000},
    {"horizon": 10, "weight": 0.36, "name": "H10_adaptive_b0.36", "steps": 150_000},
    {"horizon": 10, "weight": 0.50, "name": "H10_adaptive_b0.50", "steps": 150_000},
    {"horizon": 6, "weight": 0.74, "name": "H6_adaptive_b0.74", "steps": 150_000},
    {"horizon": 6, "weight": 1.30, "name": "H6_adaptive_b1.3", "steps": 150_000},
    {"horizon": 6, "weight": 1.65, "name": "H6_adaptive_b1.65", "steps": 150_000},
]


def self_check_adaptive():
    schedule = AdaptiveBeta(0.4)
    # Synthetic: rear directions have large error, forward have zero.
    psi = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    z = torch.stack([torch.cos(torch.tensor(psi)), torch.sin(torch.tensor(psi))], dim=-1).float()
    # Endpoints aligned in front, 60 deg off in the back half.
    ang = torch.tensor(psi, dtype=torch.float32)
    off = torch.where((ang > 0.5 * math.pi) & (ang < 1.5 * math.pi), 1.0, 0.0)
    end_ang = ang + off * (math.pi / 3)
    u = torch.stack([torch.cos(end_ang), torch.sin(end_ang)], dim=-1)
    stats = schedule.update_from_endpoints(z, u)
    assert stats["updated"]
    g = schedule.psi_grid
    beta = schedule.beta
    front = beta[(g < 0.25 * math.pi) | (g > 1.75 * math.pi)].mean()
    back = beta[(g > 0.75 * math.pi) & (g < 1.25 * math.pi)].mean()
    assert back > front, (front, back)
    assert abs(stats["mean_beta"] - 0.4) < 0.08, stats
    e = angular_error_from_u(z[:1], u[:1])
    assert 0.0 <= float(e) <= 1.0
    log(f"adaptive self-check ok  front_beta={front:.3f} back_beta={back:.3f} mean={stats['mean_beta']:.3f}")


def best_constant(rows, horizon: float):
    cands = [r for r in rows if r.get("horizon") == horizon and r.get("mode") == "dart"]
    if not cands:
        return None
    return min(cands, key=lambda r: (r["mean_err_deg"], r["max_err_deg"], r["blind_deg"]))


def diagnose(result, job, constant_ref, L):
    notes = []
    if constant_ref is not None:
        if result["mean_err_deg"] > constant_ref["mean_err_deg"] + 0.5:
            notes.append(
                f"mean_err worse than constant beta={constant_ref['weight']} "
                f"({result['mean_err_deg']:.2f} > {constant_ref['mean_err_deg']:.2f})"
            )
        if result["max_err_deg"] > constant_ref["max_err_deg"] + 2.0:
            notes.append(
                f"max_err worse than constant "
                f"({result['max_err_deg']:.2f} > {constant_ref['max_err_deg']:.2f})"
            )
        if result["blind_deg"] > constant_ref["blind_deg"] + 5.0:
            notes.append(
                f"blind worse than constant "
                f"({result['blind_deg']:.2f} > {constant_ref['blind_deg']:.2f})"
            )
    snap = result.get("adaptive")
    if snap is not None:
        beta = np.asarray(snap["beta"])
        if beta.max() - beta.min() < 0.05 * job["weight"]:
            notes.append("beta schedule barely adapted (almost flat); on-policy errors may be too uniform")
        if snap["updates"] < 3:
            notes.append(f"too few adaptive updates ({snap['updates']})")
        if L == L and job["weight"] < 0.5 * L:
            notes.append(f"beta_base={job['weight']} is far below measured L={L:.3f}; alignment bound stays large")
    if not notes:
        notes.append("no red flags vs constant baseline / schedule health")
    return notes


def wait_for_gpu(pid_guess: int | None = None):
    """Wait until the constant sweep has written all expected CSV rows."""
    expected = 20  # 2 horizons x (5 dart/metra + 3 squared + 2 norm)
    log("waiting for the constant-beta sweep to finish...")
    while True:
        csv_path = RESULTS / "sac_results.csv"
        n = 0
        if csv_path.exists():
            n = max(0, len(csv_path.read_text(encoding="utf-8").strip().splitlines()) - 1)
        if n >= expected:
            log(f"constant sweep finished ({n} rows); starting adaptive")
            return
        log(f"constant sweep progress: {n}/{expected} rows")
        time.sleep(30)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n===== {datetime.now().isoformat()} adaptive phase device={device} =====")
    self_check_adaptive()
    wait_for_gpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    oracle = load_oracle_scalars()
    existing = load_sac_rows()
    fig_dir = RESULTS / "figures"
    sac_dir = RESULTS / "sac"
    ckpt_dir = RESULTS / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sac_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "sac_results.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "name,horizon,mode,weight,max_err_deg,mean_err_deg,blind_deg,mean_forward,bound_deg,seconds\n",
            encoding="utf-8",
        )

    for job in ADAPTIVE_JOBS:
        npz_path = sac_dir / f"{job['name']}.npz"
        if npz_path.exists():
            log(f"skip existing {job['name']}")
            continue
        L = float(oracle.get(job["horizon"], {}).get("L", float("nan")))
        const_ref = best_constant(existing, job["horizon"])
        log(
            f"training {job['name']} adaptive beta_base={job['weight']:g} L={L} "
            f"steps={job['steps']}  best_constant="
            f"{None if const_ref is None else (const_ref['name'], const_ref['mean_err_deg'])}"
        )
        try:
            result = train_one(
                job["horizon"],
                "adaptive",
                job["weight"],
                device,
                total_steps=job["steps"],
                seed=0,
                L=L,
                log=log,
                adaptive_update_every=2_000,
            )
        except Exception:
            log(f"FAILED {job['name']}\n{traceback.format_exc()}")
            continue
        notes = diagnose(result, job, const_ref, L)
        for note in notes:
            log(f"  diagnose: {note}")
        actor = result.pop("actor")
        adaptive_snap = result.pop("adaptive", None)
        torch.save(
            {
                "actor": actor.state_dict(),
                "job": job,
                "bound_deg": result["bound_deg"],
                "adaptive": adaptive_snap,
            },
            ckpt_dir / f"{job['name']}.pt",
        )
        del actor
        np.savez_compressed(
            npz_path,
            u=result["u"],
            z=result["z"],
            psi=result["psi"],
            err=result["err"],
            steps_hist=np.asarray(result["history"]["steps"]),
            max_err_hist=np.asarray(result["history"]["max_err_deg"]),
            mean_err_hist=np.asarray(result["history"]["mean_err_deg"]),
            blind_hist=np.asarray(result["history"]["blind_deg"]),
            beta_hist=np.asarray(adaptive_snap["beta"]) if adaptive_snap else np.zeros(1),
            e_hat=np.asarray(adaptive_snap["e_hat"]) if adaptive_snap else np.zeros(1),
            psi_beta=np.asarray(adaptive_snap["psi"]) if adaptive_snap else np.zeros(1),
        )
        payload = {
            "name": job["name"],
            "horizon": job["horizon"],
            "mode": "adaptive",
            "weight": job["weight"],
            "max_err_deg": result["max_err_deg"],
            "mean_err_deg": result["mean_err_deg"],
            "blind_deg": result["blind_deg"],
            "blind_center_deg": result["blind_center_deg"],
            "mean_forward": result["mean_forward"],
            "bound_deg": result["bound_deg"],
            "seconds": result["seconds"],
            "history": result["history"],
            "diagnose": notes,
            "adaptive_mean_beta": None if adaptive_snap is None else float(np.mean(adaptive_snap["beta"])),
            "adaptive_updates": None if adaptive_snap is None else int(adaptive_snap["updates"]),
        }
        (sac_dir / f"{job['name']}.json").write_text(
            json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
        )
        plot_training(
            result["history"],
            result["u"],
            result["psi"],
            f"{job['name']}  max {result['max_err_deg']:.1f}  mean {result['mean_err_deg']:.1f}  blind {result['blind_deg']:.1f}",
            fig_dir / f"{job['name']}.png",
        )
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{job['name']},{job['horizon']},adaptive,{job['weight']},"
                f"{result['max_err_deg']:.4f},{result['mean_err_deg']:.4f},{result['blind_deg']:.4f},"
                f"{result['mean_forward']:.4f},{result['bound_deg']},{result['seconds']:.1f}\n"
            )
        existing.append(payload)
        log(
            f"finished {job['name']} in {result['seconds']:.1f}s  "
            f"max_err={result['max_err_deg']:.2f}  mean={result['mean_err_deg']:.2f}  "
            f"blind={result['blind_deg']:.2f}"
        )

    rows = load_sac_rows()
    if rows:
        plot_learned_summary(rows, oracle, fig_dir / "sac_vs_oracle.png")
        # Write a short comparison focused on adaptive vs best constant / metra / norm.
        lines = ["# Adaptive vs baselines", ""]
        for h in (10, 6):
            subset = [r for r in rows if r.get("horizon") == h]
            if not subset:
                continue
            lines.append(f"## H={h}")
            for r in sorted(subset, key=lambda x: x["mean_err_deg"]):
                lines.append(
                    f"- {r['name']}: mean={r['mean_err_deg']:.2f}  max={r['max_err_deg']:.2f}  blind={r['blind_deg']:.2f}"
                )
            lines.append("")
        (RESULTS / "adaptive_comparison.md").write_text("\n".join(lines), encoding="utf-8")
        log(f"wrote {RESULTS / 'adaptive_comparison.md'}")
    log("adaptive phase done")


if __name__ == "__main__":
    main()
