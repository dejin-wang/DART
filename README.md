# DART
# Dubins-Car DART

Training and evaluation code for direction-aware skill learning on a Dubins car
(METRA baseline, constant-β DART, and direction-adaptive DART).

This release contains **only** the Dubins-car stack: no other environments and
**no pretrained checkpoints or paper result tables**. Train locally, then run
the eval scripts on the checkpoints you produce.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA build of PyTorch is recommended for full training runs.

## Repository layout

```
dubins_car/          # environment, dynamics, SAC trainer, AdaptiveBeta
run_all.py           # oracle L + constant METRA/DART/squared/norm training
run_adaptive.py      # direction-adaptive DART (η=1)
run_eta2.py          # adaptive ablation with η=2
run_seed_suite.py    # multi-seed constant + adaptive suite
run_beta_sweep.py    # denser constant-β grid
eval_goal_reaching.py
eval_adr.py          # distance-adaptive replanning controller
eval_dubins_followups.py
```

Training and evaluation write outputs under `results/` (created automatically).

## Quick start (single seed)

### 1) Oracle reference \(L\) (optional but useful)

```bash
python run_all.py --phase oracle
```

### 2) Train constant objectives (METRA / DART / …)

```bash
python run_all.py --phase train
```

Checkpoints land in `results/checkpoints/` (seed 0) or `results/seed{k}/checkpoints/`.

### 3) Train direction-adaptive DART

```bash
python run_adaptive.py
```

Paper-style multi-seed suite (constant + adaptive):

```bash
python run_seed_suite.py --seeds 0,1,2
```

Extra constant-β sweep (for β-sensitivity plots):

```bash
python run_beta_sweep.py
```

### 4) Evaluate

Goal reaching / replanning:

```bash
python eval_goal_reaching.py
python eval_adr.py
```

Controller ablation, random-start skill metrics, β-sweep metrics, replan counts
(expects the checkpoints named in each script’s policy tables):

```bash
python eval_dubins_followups.py
```

## Notes

- Horizons in the scripts are in units of the minimum turning radius
  (\(h=6\) and \(h=10\) are the main settings).
- Adaptive updates use `AdaptiveBeta` (`dubins_car/adaptive.py`); the default
  training path uses \(\eta=1\). Use `run_eta2.py` for the \(\eta=2\) ablation.
- `d3_schedule.py` is present because the SAC trainer imports it; the main
  paper path does not require training a D3 schedule.

## Citation

If you use this code, please cite the associated paper (add your BibTeX here).
