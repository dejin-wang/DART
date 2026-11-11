# -*- coding: utf-8 -*-
"""Dubins-car check of METRA's blind sector and the absolute-penalty bound.

The car drives forward at unit speed with turning radius 1. phi(s) = (x, y)
is fixed. A horizon of 10 turning radii is 200 steps of dt=0.05; 6 radii is
120 steps. Ideal endpoints are computed by direct maximization over action
sequences. SAC then trains METRA (beta=0), the absolute lateral penalty at the
requested weights below and above the reference L, a squared lateral penalty,
and norm matching F = (1+lambda) z·u - lambda ||u||.
"""

from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dubins_car.dynamics import (  # noqa: E402
    DT,
    V,
    horizon_steps,
    integrate_step_np,
    self_test,
    terminal_objective_np,
)
from dubins_car.env import DubinsCarEnv  # noqa: E402
from dubins_car.oracle import ideal_skills_for_horizon  # noqa: E402
from dubins_car.plots import (  # noqa: E402
    plot_error_vs_bound,
    plot_learned_summary,
    plot_oracle,
    plot_other_penalties,
    plot_radial,
    plot_training,
)
from dubins_car.torch_dyn import rollout_actions_torch  # noqa: E402
from dubins_car.train_sac import train_one  # noqa: E402

REF = {
    10: {"L": 0.32, "blind": 33.0, "betas": [0.0, 0.16, 0.28, 0.36, 0.64]},
    6: {"L": 1.48, "blind": 70.0, "betas": [0.0, 0.74, 1.30, 1.65, 2.96]},
}
SQUARED_TRAIN = [0.05, 0.2, 1.0]
NORM_TRAIN = [1.0, 5.0]
SQUARED_ORACLE = [0.01, 0.05, 0.2, 1.0, 5.0]
NORM_ORACLE = [0.25, 1.0, 5.0, 20.0]


def log(msg: str) -> None:
    print(msg, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "run.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        value = float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, float):
        value = obj
    else:
        raise TypeError(type(obj))
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def dense_betas(radii: float) -> list[float]:
    if radii >= 10:
        grid = np.linspace(0.0, 1.0, 21)
    else:
        grid = np.linspace(0.0, 3.2, 33)
    values = np.concatenate([grid, np.asarray(REF[radii]["betas"], dtype=np.float64)])
    return [float(v) for v in np.round(values, 5)]


def scalar_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, dict):
            continue
        out[key] = value
    return out


def test_all(device) -> None:
    self_test()
    actions = torch.ones(1, 1, device=device)
    xy, th = rollout_actions_torch(actions, dt=math.pi / 2)
    if not torch.allclose(xy.cpu(), torch.tensor([[1.0, 1.0]]), atol=1e-5):
        raise AssertionError(f"quarter circle xy={xy}")
    if abs(float(th) - math.pi / 2) > 1e-5:
        raise AssertionError(f"quarter circle heading={float(th)}")

    rng = np.random.default_rng(1)
    state = np.zeros(3)
    tstate = torch.zeros(1, 3, device=device)
    for _ in range(50):
        a = float(rng.uniform(-1, 1))
        state = integrate_step_np(state, a)
        tstate = __import__("dubins_car.torch_dyn", fromlist=["step_dynamics_torch"]).step_dynamics_torch(
            tstate, torch.tensor([a], device=device)
        )
    if np.max(np.abs(state - tstate.cpu().numpy()[0])) > 1e-4:
        raise AssertionError("numpy and torch dynamics diverged")

    env = DubinsCarEnv(horizon_radii=10, reward_mode="dart", weight=0.35)
    env.reset(seed=0, options={"z": np.array([0.6, 0.8])})
    total = 0.0
    terminated = False
    info = {}
    while not terminated:
        _, reward, terminated, _, info = env.step(env.action_space.sample())
        total += reward
    got = terminal_objective_np(info["u"], info["z"], "dart", 0.35)
    if abs(total - got) > 1e-5:
        raise AssertionError(f"env telescope {total} != {got}")

    env = DubinsCarEnv(horizon_radii=10, reward_mode="metra")
    env.reset(seed=0, options={"z": np.array([1.0, 0.0])})
    info = {}
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    if abs(info["u"][0] - 10.0) > 1e-5 or abs(info["u"][1]) > 1e-5:
        raise AssertionError(f"straight skill ended at {info['u']}")
    assert horizon_steps(10) == 200 and abs(200 * DT * V - 10.0) < 1e-12
    log("self-tests passed")


def save_oracle(summary: dict) -> None:
    radii = int(round(summary["horizon_radii"]))
    tag = f"H{radii:g}"
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_oracle(summary, REF[radii]["betas"], REF[radii]["L"], REF[radii]["blind"], fig_dir / f"oracle_{tag}.png")
    plot_radial(summary, fig_dir / f"radial_{tag}.png")
    plot_other_penalties(summary, fig_dir / f"penalties_{tag}.png")
    plot_error_vs_bound(summary, fig_dir / f"error_vs_bound_{tag}.png")
    np.savez_compressed(
        RESULTS / f"oracle_{tag}.npz",
        psi=summary["psi"],
        rho=summary["rho"],
        dlog_smooth=summary["dlog_smooth"],
        dlog_raw=summary["dlog_raw"],
        z=summary["z"],
        cloud=summary["cloud"],
        u=np.stack([r["u"] for r in summary["results"]]),
        err=np.stack([r["err"] for r in summary["results"]]),
        weight=np.array([r["weight"] for r in summary["results"]], dtype=np.float64),
        mode=np.array([r["mode"] for r in summary["results"]]),
        L=np.float64(summary["L"]),
        L_raw=np.float64(summary["L_raw"]),
    )
    payload = {
        "horizon_radii": summary["horizon_radii"],
        "steps": summary["steps"],
        "L": summary["L"],
        "L_raw": summary["L_raw"],
        "reference_L": REF[radii]["L"],
        "reference_blind_deg": REF[radii]["blind"],
        "east_projection": summary["east_projection"],
        "star_median_err": summary["star_median_err"],
        "star_frac": summary["star_frac"],
        "radial_lat_max": summary["radial_lat_max"],
        "results": [scalar_row(r) for r in summary["results"]],
    }
    (RESULTS / f"oracle_{tag}.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    metra = next(r for r in summary["results"] if r["mode"] == "dart" and r["weight"] == 0.0)
    log(
        f"oracle {tag}: measured L={summary['L']:.3f} (raw {summary['L_raw']:.3f}), "
        f"reference L={REF[radii]['L']}. METRA blind={metra['blind_deg']:.1f} deg "
        f"centered at {metra['blind_center_deg']:.1f} deg "
        f"(reference about {REF[radii]['blind']:.0f})."
    )


def training_jobs():
    jobs = []
    for radii, spec in REF.items():
        for beta in spec["betas"]:
            if beta == 0.0:
                jobs.append({"horizon": radii, "mode": "metra", "weight": 0.0, "name": f"H{radii:g}_metra"})
            else:
                jobs.append(
                    {
                        "horizon": radii,
                        "mode": "dart",
                        "weight": beta,
                        "name": f"H{radii:g}_dart_b{beta:g}",
                    }
                )
        for lam in SQUARED_TRAIN:
            jobs.append(
                {"horizon": radii, "mode": "squared", "weight": lam, "name": f"H{radii:g}_squared_{lam:g}"}
            )
        for lam in NORM_TRAIN:
            jobs.append({"horizon": radii, "mode": "norm", "weight": lam, "name": f"H{radii:g}_norm_{lam:g}"})
    return jobs


def load_oracle_scalars():
    out = {}
    for path in RESULTS.glob("oracle_H*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[int(round(payload["horizon_radii"]))] = payload
    return out


def load_sac_rows():
    rows = []
    sac_dir = RESULTS / "sac"
    if not sac_dir.exists():
        return rows
    for path in sorted(sac_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "test", "oracle", "train"], default="all")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--polish-steps", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    log(
        f"\n===== {datetime.now().isoformat()} phase={args.phase} steps={args.steps} "
        f"device={device} torch={torch.__version__} ====="
    )
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    if args.phase in ("all", "test"):
        test_all(device)
        if args.phase == "test":
            return

    oracle = {}
    if args.phase in ("all", "oracle"):
        for radii in (10, 6):
            t0 = time.time()
            log(f"computing ideal skills for horizon {radii:g}")
            summary = ideal_skills_for_horizon(
                radii,
                dense_betas(radii),
                SQUARED_ORACLE,
                NORM_ORACLE,
                device,
                seed=args.seed,
                polish_steps=args.polish_steps,
            )
            save_oracle(summary)
            oracle[radii] = {
                "L": summary["L"],
                "results": [scalar_row(r) for r in summary["results"]],
            }
            log(f"horizon {radii:g} oracle finished in {time.time() - t0:.1f}s")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        oracle = load_oracle_scalars()

    if args.phase not in ("all", "train"):
        return

    if not oracle:
        oracle = load_oracle_scalars()
    fig_dir = RESULTS / "figures"
    sac_dir = RESULTS / "sac"
    ckpt_dir = RESULTS / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sac_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "sac_results.csv"
    if args.force or not csv_path.exists():
        csv_path.write_text(
            "name,horizon,mode,weight,max_err_deg,mean_err_deg,blind_deg,mean_forward,bound_deg,seconds\n",
            encoding="utf-8",
        )

    for job in training_jobs():
        npz_path = sac_dir / f"{job['name']}.npz"
        if npz_path.exists() and not args.force:
            log(f"skip existing {job['name']}")
            continue
        L = float(oracle.get(job["horizon"], {}).get("L", float("nan")))
        log(f"training {job['name']}  mode={job['mode']} weight={job['weight']:g} L={L}")
        try:
            result = train_one(
                job["horizon"],
                job["mode"],
                job["weight"],
                device,
                total_steps=args.steps,
                seed=args.seed,
                L=L,
                log=log,
            )
        except Exception:
            log(f"FAILED {job['name']}\n{traceback.format_exc()}")
            continue
        actor = result.pop("actor")
        torch.save(
            {"actor": actor.state_dict(), "job": job, "bound_deg": result["bound_deg"]},
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
        )
        payload = {
            "name": job["name"],
            "horizon": job["horizon"],
            "mode": job["mode"],
            "weight": job["weight"],
            "max_err_deg": result["max_err_deg"],
            "mean_err_deg": result["mean_err_deg"],
            "blind_deg": result["blind_deg"],
            "blind_center_deg": result["blind_center_deg"],
            "mean_forward": result["mean_forward"],
            "bound_deg": result["bound_deg"],
            "seconds": result["seconds"],
            "history": result["history"],
        }
        (sac_dir / f"{job['name']}.json").write_text(
            json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
        )
        plot_training(
            result["history"],
            result["u"],
            result["psi"],
            f"{job['name']}  max err {result['max_err_deg']:.1f} deg  blind {result['blind_deg']:.1f} deg",
            fig_dir / f"{job['name']}.png",
        )
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{job['name']},{job['horizon']},{job['mode']},{job['weight']},"
                f"{result['max_err_deg']:.4f},{result['mean_err_deg']:.4f},{result['blind_deg']:.4f},"
                f"{result['mean_forward']:.4f},{result['bound_deg']},{result['seconds']:.1f}\n"
            )
        log(
            f"finished {job['name']} in {result['seconds']:.1f}s  "
            f"max_err={result['max_err_deg']:.2f}  blind={result['blind_deg']:.2f}  "
            f"bound={result['bound_deg']}"
        )

    rows = load_sac_rows()
    if rows:
        if not oracle:
            oracle = load_oracle_scalars()
        plot_learned_summary(rows, oracle, fig_dir / "sac_vs_oracle.png")
        log(f"wrote {fig_dir / 'sac_vs_oracle.png'}")
    log("done")


if __name__ == "__main__":
    main()
