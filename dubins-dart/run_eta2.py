# -*- coding: utf-8 -*-
"""Train adaptive DART with eta=2 (stronger error emphasis than eta=1).

eta=0 -> uniform beta (constant). eta=1 -> prior adaptive runs.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from dubins_car.train_sac import train_one  # noqa: E402
from run_adaptive import ADAPTIVE_JOBS  # noqa: E402
from run_all import REF, json_default, load_oracle_scalars, log  # noqa: E402

ETA = 2.0


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--paper-only",
        action="store_true",
        help="Only H10 beta0=0.50 and H6 beta0=1.30 (table rows).",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    seed = int(args.seed)
    eta = float(args.eta)
    out = RESULTS / f"eta{eta:g}_seed{seed}"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sac_results.csv"
    log_path = out / "run.log"
    oracles = load_oracle_scalars()

    jobs = list(ADAPTIVE_JOBS)
    if args.paper_only:
        jobs = [j for j in jobs if j["name"] in ("H10_adaptive_b0.50", "H6_adaptive_b1.3")]

    def slog(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        log(msg)

    fields = [
        "name", "horizon", "mode", "weight", "eta", "seed",
        "max_err_deg", "mean_err_deg", "blind_deg", "mean_forward",
        "seconds", "steps", "mean_beta", "beta_span",
    ]
    slog(f"=== adaptive eta={eta:g} seed={seed} device={device} jobs={len(jobs)} ===")
    t0 = time.time()
    done, failed = [], []

    for i, job in enumerate(jobs, 1):
        name = f"{job['name']}_eta{eta:g}"
        ckpt = ckpt_dir / f"{name}.pt"
        if ckpt.exists():
            slog(f"[{i}/{len(jobs)}] SKIP {name}")
            done.append(name)
            continue
        radii = float(job["horizon"])
        L = float(oracles.get(int(radii), {}).get("L", REF[int(radii)]["L"]))
        slog(f"[{i}/{len(jobs)}] TRAIN {name} beta_base={job['weight']} steps={job['steps']}")
        try:
            result = train_one(
                horizon_radii=radii,
                mode="adaptive",
                weight=float(job["weight"]),
                device=device,
                total_steps=int(job["steps"]),
                num_envs=args.num_envs,
                updates_per_step=args.num_envs,
                seed=seed,
                L=L,
                log=slog,
                eta=eta,
            )
            snap = result.get("adaptive") or {}
            beta = snap.get("beta")
            if beta is not None:
                import numpy as np

                beta = np.asarray(beta)
                mean_beta = float(beta.mean())
                beta_span = float(beta.max() - beta.min())
            else:
                mean_beta = float("nan")
                beta_span = float("nan")
            torch.save(
                {
                    "actor": result["actor"].state_dict(),
                    "horizon": radii,
                    "mode": "adaptive",
                    "weight": float(job["weight"]),
                    "eta": eta,
                    "seed": seed,
                    "adaptive": snap,
                    "metrics": {
                        k: result[k]
                        for k in ("max_err_deg", "mean_err_deg", "blind_deg", "mean_forward", "seconds")
                    },
                },
                ckpt,
            )
            row = {
                "name": name,
                "horizon": radii,
                "mode": "adaptive",
                "weight": float(job["weight"]),
                "eta": eta,
                "seed": seed,
                "max_err_deg": result["max_err_deg"],
                "mean_err_deg": result["mean_err_deg"],
                "blind_deg": result["blind_deg"],
                "mean_forward": result["mean_forward"],
                "seconds": result["seconds"],
                "steps": int(job["steps"]),
                "mean_beta": mean_beta,
                "beta_span": beta_span,
            }
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            (out / f"{name}.json").write_text(
                json.dumps({**row, "adaptive": snap}, indent=2, default=json_default),
                encoding="utf-8",
            )
            slog(
                f"  FINISHED {name} mean={result['mean_err_deg']:.2f} max={result['max_err_deg']:.2f} "
                f"blind={result['blind_deg']:.2f} beta_span={beta_span:.3f}"
            )
            done.append(name)
        except Exception:
            slog(f"  FAILED {name}\n{traceback.format_exc()}")
            failed.append(name)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    slog(
        f"=== eta={eta:g} seed={seed} done ok={len(done)}/{len(jobs)} failed={failed} "
        f"wall={(time.time()-t0)/3600:.2f}h ==="
    )


if __name__ == "__main__":
    main()
