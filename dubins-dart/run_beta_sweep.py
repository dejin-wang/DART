# -*- coding: utf-8 -*-
"""Train missing constant-DART betas for theory–experiment β sweep."""

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

import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from dubins_car.train_sac import train_one  # noqa: E402

# Suggested grid; skip if checkpoint already exists under results/checkpoints.
BETAS = {
    10: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    6: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100_000)
    args = ap.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out = RESULTS / ("checkpoints" if args.seed == 0 else f"seed{args.seed}/checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS / "beta_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed{args.seed}_run.log"
    csv_path = log_dir / f"seed{args.seed}_results.csv"

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    fields = [
        "name",
        "horizon",
        "beta",
        "seed",
        "max_err_deg",
        "mean_err_deg",
        "blind_deg",
        "mean_forward",
        "seconds",
        "steps",
    ]
    jobs = [(H, b) for H, bs in BETAS.items() for b in bs]
    log(f"=== beta sweep seed={args.seed} device={device} jobs={len(jobs)} ===")
    for i, (H, beta) in enumerate(jobs, 1):
        name = f"H{H}_dart_b{beta:g}".replace(".", "p") if False else (
            f"H{H}_dart_b{beta}" if beta != int(beta) else f"H{H}_dart_b{int(beta)}"
        )
        # Match existing naming: H10_dart_b0.64, H6_dart_b1.3
        if abs(beta - round(beta)) < 1e-9:
            name = f"H{H}_dart_b{int(round(beta))}"
        else:
            name = f"H{H}_dart_b{beta:g}"
        ckpt = out / f"{name}.pt"
        if ckpt.exists():
            log(f"[{i}/{len(jobs)}] SKIP {name}")
            continue
        log(f"[{i}/{len(jobs)}] TRAIN {name}")
        try:
            result = train_one(
                horizon_radii=H,
                mode="dart",
                weight=float(beta),
                device=device,
                total_steps=args.steps,
                seed=args.seed,
                log=log,
            )
            torch.save(
                {
                    "actor": result["actor"].state_dict(),
                    "horizon": H,
                    "mode": "dart",
                    "weight": float(beta),
                },
                ckpt,
            )
            row = {
                "name": name,
                "horizon": H,
                "beta": beta,
                "seed": args.seed,
                "max_err_deg": result["max_err_deg"],
                "mean_err_deg": result["mean_err_deg"],
                "blind_deg": result["blind_deg"],
                "mean_forward": result["mean_forward"],
                "seconds": result["seconds"],
                "steps": args.steps,
            }
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            (log_dir / f"seed{args.seed}_{name}.json").write_text(
                json.dumps({**row, "history": result["history"]}, indent=2, allow_nan=True),
                encoding="utf-8",
            )
            log(
                f"  FINISHED {name} mean={result['mean_err_deg']:.2f} "
                f"blind={result['blind_deg']:.2f} fwd={result['mean_forward']:.2f}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            log(f"  FAILED {name}\n{traceback.format_exc()}")
    log("=== beta sweep done ===")


if __name__ == "__main__":
    main()
