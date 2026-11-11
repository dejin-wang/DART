# DART

Code for **DART**, a direction-aware extension of METRA for unsupervised skill discovery.
DART keeps METRA's representation objective and adds a penalty on lateral deviation from the
skill direction. For skill $z$ (a unit vector) and $u_t = \phi(s_t) - \phi(s_0)$, the trajectory
return is

$$z^\top u_T - \beta\, b(u_T), \qquad b(u) = \lVert u - (z^\top u)\,z \rVert ,$$

with either a constant penalty $\beta$ or a direction-adaptive $\beta(\psi)$ that puts more weight
on skill angles with larger end-direction errors.

This repository has two independent code stacks, one for each set of experiments in the paper:

| Folder | Environments | Methods | Setup |
|---|---|---|---|
| [`dubins-dart/`](dubins-dart) | Dubins car | METRA, constant-β DART, adaptive DART | Python + PyTorch (`pip install -r requirements.txt`) |
| [`Mujoco/`](Mujoco) | Ant, HalfCheetah (state-based) | METRA, constant-β DART, adaptive DART, Norm Matching, D3 | Python 3.8 + MuJoCo 2.1, built on the official METRA code |

Each folder has its own README with installation, training and evaluation commands.
No pretrained checkpoints or result tables are included: train locally, then run the
evaluation scripts on the checkpoints you produce.

## Quick start

**Dubins car**

```bash
cd dubins-dart
pip install -r requirements.txt
python run_all.py --phase train     # METRA / constant-β DART
python run_adaptive.py              # direction-adaptive DART
python eval_goal_reaching.py
```

**Ant / HalfCheetah**

```bash
cd Mujoco
conda create --name dart python=3.8 && conda activate dart
pip install -r requirements.txt --no-deps
pip install -e . && pip install -e garaged
python train.py --env ant --method dart --seed 0      # --method: metra | dart | dart_adaptive | nm | d3
```

See [`dubins-dart/README.md`](dubins-dart/README.md) and [`Mujoco/README.md`](Mujoco/README.md)
for the full experiment pipelines and evaluation protocols.

## Acknowledgements

The MuJoCo code builds on [METRA](https://github.com/seohongpark/METRA), which is based on
[LSD](https://github.com/seohongpark/LSD) and [garage](https://github.com/rlworkgroup/garage);
see `Mujoco/LICENSE_garage`. Norm Matching follows Atanassov et al. (2024) and the D3 schedule
follows Cathomen et al. (CoRL 2025).

## Citation

If you use this code, please cite the associated paper (BibTeX coming soon).
