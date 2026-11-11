# -*- coding: utf-8 -*-
"""Auto-run a full Dubins training seed (METRA/DART/squared/norm/adaptive).

Why the first sweep felt slow
-----------------------------
Default train_one uses num_envs=16 and updates_per_step=16 (UTD≈1).
Each outer step does 16 sequential small-MLP SAC updates; measured ~95
environment-steps/s → ~22 min per 1e5-step job, ~26 min per adaptive 1.5e5.
A full paper sweep is ~26 jobs ≈ 10 h wall-clock, not because the car is hard,
but because UTD=1 × many sequential jobs.

This script runs the complete necessary suite for one seed and skips finished
checkpoints so it can be resumed.
"""

from __future__ import annotations

import argparse
import csv
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

from dubins_car.plots import plot_learned_summary, plot_training  # noqa: E402
from dubins_car.train_sac import train_one  # noqa: E402
from run_adaptive import ADAPTIVE_JOBS, diagnose, best_constant  # noqa: E402
from run_all import (  # noqa: E402
    REF,
    SQUARED_TRAIN,
    NORM_TRAIN,
    json_default,
    load_oracle_scalars,
    load_sac_rows,
    log,
    training_jobs,
)


def suite_jobs():
    """Full paper Dubins suite (same as original sweep + adaptive)."""
    jobs = []
    for j in training_jobs():
        jobs.append({**j, "steps": 100_000, "kind": "const"})
    for j in ADAPTIVE_JOBS:
        jobs.append(
            {
                "horizon": j["horizon"],
                "mode": "adaptive",
                "weight": j["weight"],
                "name": j["name"],
                "steps": j["steps"],
                "kind": "adaptive",
            }
        )
    return jobs


def append_csv(path: Path, row: dict, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=32, help="default was 16; 32 ~ faster wall time")
    ap.add_argument(
        "--updates-per-step",
        type=int,
        default=None,
        help="default=num_envs (UTD~1). Lower = faster, possibly weaker.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="comma substring filter, e.g. adaptive,H6")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    seed = int(args.seed)
    out_dir = RESULTS / f"seed{seed}"
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sac_results.csv"
    status_path = out_dir / "suite_status.json"
    log_path = out_dir / "run.log"

    def slog(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Also mirror into results/run.log for familiarity
    def both(msg: str):
        slog(msg)
        log(msg)

    jobs = suite_jobs()
    if args.only.strip():
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        jobs = [j for j in jobs if any(k.lower() in j["name"].lower() or k.lower() in j["mode"].lower() for k in keys)]

    oracles = load_oracle_scalars()
    fields = [
        "name", "horizon", "mode", "weight", "seed", "max_err_deg", "mean_err_deg",
        "blind_deg", "mean_forward", "bound_deg", "seconds", "steps",
    ]

    both(
        f"=== seed suite start seed={seed} device={device} jobs={len(jobs)} "
        f"num_envs={args.num_envs} updates_per_step={args.updates_per_step} ==="
    )
    both(
        "Note: previous ~22 min/job came from UTD~1 (16 SAC updates per 16 env steps) "
        "at ~95 steps/s x 1e5 steps. Not env complexity."
    )

    t_suite = time.time()
    done = []
    failed = []

    for i, job in enumerate(jobs, 1):
        name = job["name"]
        ckpt = ckpt_dir / f"{name}.pt"
        if ckpt.exists():
            both(f"[{i}/{len(jobs)}] SKIP {name} (checkpoint exists)")
            done.append(name)
            continue

        radii = float(job["horizon"])
        L = float(oracles.get(int(radii), {}).get("L", REF[int(radii)]["L"]))
        mode = job["mode"]
        weight = float(job["weight"])
        steps = int(job["steps"])
        ups = args.updates_per_step if args.updates_per_step is not None else args.num_envs

        both(
            f"[{i}/{len(jobs)}] TRAIN {name}  mode={mode} w={weight} steps={steps} L={L:.3f}"
        )
        status_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "current": name,
                    "index": i,
                    "total": len(jobs),
                    "done": done,
                    "failed": failed,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            result = train_one(
                horizon_radii=radii,
                mode=mode,
                weight=weight,
                device=device,
                total_steps=steps,
                num_envs=args.num_envs,
                updates_per_step=ups,
                seed=seed,
                L=L,
                log=slog,
            )
            torch.save(
                {
                    "actor": result["actor"].state_dict(),
                    "horizon": radii,
                    "mode": mode,
                    "weight": weight,
                    "seed": seed,
                    "metrics": {
                        k: result[k]
                        for k in (
                            "max_err_deg", "mean_err_deg", "blind_deg",
                            "mean_forward", "bound_deg", "seconds",
                        )
                    },
                    "adaptive": result.get("adaptive"),
                },
                ckpt,
            )
            # plots
            try:
                plot_training(result["history"], fig_dir / f"train_{name}.png")
                plot_learned_summary(result, L, fig_dir / f"learned_{name}.png")
            except Exception as e:
                slog(f"  plot warn: {e}")

            row = {
                "name": name,
                "horizon": radii,
                "mode": mode,
                "weight": weight,
                "seed": seed,
                "max_err_deg": result["max_err_deg"],
                "mean_err_deg": result["mean_err_deg"],
                "blind_deg": result["blind_deg"],
                "mean_forward": result["mean_forward"],
                "bound_deg": result["bound_deg"],
                "seconds": result["seconds"],
                "steps": steps,
            }
            append_csv(csv_path, row, fields)
            (out_dir / f"{name}.json").write_text(
                json.dumps({**row, "adaptive": result.get("adaptive")}, indent=2, default=json_default),
                encoding="utf-8",
            )

            if mode == "adaptive":
                # light diagnosis vs constant twin if available in seed0 or this seed
                rows = []
                if csv_path.exists():
                    import csv as _csv
                    with csv_path.open(encoding="utf-8") as f:
                        rows = list(_csv.DictReader(f))
                for r in rows:
                    r["horizon"] = float(r["horizon"])
                    r["weight"] = float(r["weight"])
                    for k in ("max_err_deg", "mean_err_deg", "blind_deg"):
                        r[k] = float(r[k])
                cref = best_constant(rows, radii)
                notes = diagnose(result, job, cref, L)
                if notes:
                    both(f"  adaptive notes: {'; '.join(notes)}")

            both(
                f"  FINISHED {name} in {result['seconds']:.1f}s  "
                f"mean={result['mean_err_deg']:.2f} max={result['max_err_deg']:.2f} "
                f"blind={result['blind_deg']:.2f}"
            )
            done.append(name)
        except Exception:
            both(f"  FAILED {name}\n{traceback.format_exc()}")
            failed.append(name)
            # continue remaining jobs
            continue

        # free GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t_suite
    summary = {
        "seed": seed,
        "elapsed_min": elapsed / 60.0,
        "done": done,
        "failed": failed,
        "n_jobs": len(jobs),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    status_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    both(
        f"=== suite done seed={seed}  {len(done)}/{len(jobs)} ok  "
        f"failed={failed}  wall={elapsed/3600:.2f} h ==="
    )


if __name__ == "__main__":
    main()
