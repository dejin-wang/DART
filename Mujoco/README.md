# DART

Code for training and evaluating DART on state-based **Ant** and **HalfCheetah**, together with the baselines used in
the paper (METRA, Norm Matching, and the D3 schedule). No trained models or results are included; every number and
figure in the paper's Ant and HalfCheetah sections can be regenerated with the commands below.

The code builds on the official [METRA](https://github.com/seohongpark/METRA) implementation (itself based on
[LSD](https://github.com/seohongpark/LSD) and [garage](https://github.com/rlworkgroup/garage)). DART only adds a
reward mode and its evaluation; METRA's representation objective, constraint and SAC settings are unchanged.

## What DART changes

For skill $z$ (a unit vector) and $u_t = \phi(s_t) - \phi(s_0)$, the SAC reward becomes

$$r_t = z^\top(\phi(s_{t+1}) - \phi(s_t)) - \beta\,[\,b(u_{t+1}) - b(u_t)\,], \qquad b(u) = \lVert u - (z^\top u)\,z \rVert ,$$

i.e. METRA's reward minus the increase of the lateral deviation from the skill ray; summed over a trajectory it is
$z^\top u_T - \beta\, b(u_T)$. The policy and Q-functions receive $(s_t, u_t, z)$. Implementation:
`iod/metra.py` (`--reward_mode ray`) and `garagei/torch/policies/ray_policy_ex.py`.

* **Constant penalty** (`--ray_beta 0.5`): the penalty is off for the first 10% of epochs and ramps linearly to its
  full value at 20% (`--ray_warmup 0.1 0.2`).
* **Adaptive penalty** (`--ray_beta_mode adaptive`): $\beta(\psi) = \beta_{\mathrm{base}}\,(1+\hat e(\psi))^\eta /
  \mathbb{E}_\psi[(1+\hat e)^\eta]$ over the skill angle $\psi$, with $\hat e$ the end-angle errors
  $e = (1-\cos\angle(u_T, z))/2$ of the current epoch's training trajectories, interpolated on 3600 angular bins and
  updated every epoch ($\eta = 1$, `--ray_adapt_eta`; $\beta_{\mathrm{base}} = 0.5$).

## Installation

Python 3.8 and [MuJoCo 2.1](https://github.com/deepmind/mujoco/releases/tag/2.1.0) (extracted to `~/.mujoco/mujoco210`).

```bash
conda create --name dart python=3.8
conda activate dart
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin   # train.py also sets this
```

Check the adaptive-penalty code with the CPU unit tests: `python tests/test_ray_adaptive.py`.

## Training

```bash
python train.py --env ant --method dart --seed 0
```

| `--method` | Method | Settings |
|---|---|---|
| `metra` | METRA | as in the METRA paper (2-D continuous skills) |
| `dart` | DART, constant $\beta$ | $\beta = 0.5$ |
| `dart_adaptive` | DART, adaptive $\beta(\psi)$ | $\beta_{\mathrm{base}} = 0.5$, $\eta = 1$ |
| `nm` | Norm Matching | skill radius $R = 24$ (per-step target $0.12\,z$) |
| `d3` | D3 schedule | METRA → norm matching as $\cos\angle(\Delta\phi, z)$ goes from 0.5 to 0.7 |

`--env` is `ant` (29-D state) or `half_cheetah` (18-D state). All runs use METRA's state-based setup: 2-D skills,
10001 epochs of 8 trajectories × 200 steps (1.6 × 10⁷ environment steps), one GPU. Checkpoints and logs go to
`exp/<env>_<method>/sd<seed>_<timestamp>_.../`. Extra arguments are passed to `tests/main.py`, e.g.
`--ray_beta 0.8`. The paper uses seeds 0–3 on Ant (0–1 for adaptive DART) and seeds 0–1 on HalfCheetah.

## Evaluation

All evaluations use the final checkpoint (`--epoch 10000`), run on CPU, and write to `results/`. Name each output
`<method>_s<seed>` (method as in the table above); the summary scripts collect all seeds of each method.

### Ant

```bash
RUN=exp/ant_dart/sd000_...     # a training run directory
python eval/ant/rollout_2000.py      --run $RUN --label dart_s0   # 2000 evenly spaced skills -> directional consistency
python eval/ant/goal_reach.py        --run $RUN --label dart_s0   # moving-start goals + AntMultiGoals (ADR)
python eval/ant/ant_multigoals_hrl.py --run $RUN --label dart_s0  # AntMultiGoals with a trained high-level SAC
python eval/ant/rollout_randstart.py --run $RUN --label dart_s0   # random start poses, at rest and mid-gait
python eval/ant/summarize.py         # -> results/ant/summary.md, results/ant/fig_ant.png
```

* **Directional consistency:** latent error $\angle(u_T, z)$; physical error between the x–y displacement and $Qz$
  ($Q$ the best orthogonal map from skill space to the floor); largest angular gap between neighbouring skills'
  end directions; distance; state coverage.
* **Goal reaching (moving start):** after a 50-step walk, 256 goals 10–40 m away in all directions relative to the
  motion, 400 steps, 3 m tolerance. The skill is $z = \frac{\phi(g)-\phi(s)}{\lVert\phi(g)-\phi(s)\rVert}$, re-selected
  every step ($h=1$), every 25 steps ($h=25$), or by Distance-Adaptive Replanning (ADR), which holds $z$ for
  $h_k = \min(25, \lfloor (D_k - r_g)_+ / (2\hat v_k) \rfloor)$ steps and stops inside the goal ball. The stop /
  no-stop ablation (`bearing1_stop`, `bearing25_stop`, `adr_nostop`) runs by default.
* **AntMultiGoals** (METRA App. F): four goals in turn within ±7.5 m, 50 steps each, return 2.5 per goal (max 10).

### HalfCheetah

```bash
RUN=exp/half_cheetah_dart/sd000_...
python eval/half_cheetah/hc_protocol_eval.py    --run $RUN --label dart_s0             # exploration, behaviour, HalfCheetahGoal (h=1)
python eval/half_cheetah/hc_goal_controllers.py --run $RUN --label dart_s0 --mode free # h=1 / h=25 / ADR
python eval/half_cheetah/hc_goal_controllers.py --run $RUN --label dart_s0 --mode sac  # high-level SAC
python eval/half_cheetah/summarize.py   # -> results/half_cheetah/summary.md, results/half_cheetah/fig_hc.png
```

* **Exploration:** 360 evenly spaced skills, 200 steps: mean $|x_T|$, visited 1 m x-bins, x range.
* **HalfCheetahGoal** (METRA App. F.1): 256 goals uniform in [−100, 100] m, 3 m tolerance, 200 steps, identical goals
  and start states for all methods; reach = success at any step, final = success at the last step.
* **Behaviour:** flipping, spinning, stationary skills, mode entropy, posture diversity, and how much of the 2-D latent
  endpoint is explained by x-motion vs. torso pitch.

## Repository layout

```
train.py                  training launcher (all methods, both environments)
tests/main.py             training entry point (METRA's, with the DART / NM / D3 reward modes)
iod/metra.py              METRA + DART (ray reward, adaptive beta), norm matching, D3
garagei/torch/policies/ray_policy_ex.py   policy input (s, u_t, z) for DART
eval/ant/                 Ant evaluations and summary
eval/half_cheetah/        HalfCheetah evaluations and summary
envs/, garagei/, garaged/, dowel/        environments and libraries from METRA / garage
```

METRA's pixel-based Kitchen environment (its `lexa` dependency) is not included; the other METRA environments are
left in place but were not used for DART.

## License and acknowledgements

This code is derived from METRA, LSD and garage; see `LICENSE_garage` for garage's license, and the METRA repository
for its terms. Norm Matching follows Atanassov et al. (2024) and the D3 schedule follows Cathomen et al. (CoRL 2025).
