#!/usr/bin/env python3
"""Train skills on state-based Ant or HalfCheetah: METRA, DART (constant or adaptive lateral penalty), and the
norm-matching / D3 baselines, all with the same settings (METRA's state-based setup, 2-D continuous skills).

    python train.py --env ant --method dart --seed 0
    python train.py --env half_cheetah --method dart_adaptive --seed 1
    python train.py --env ant --method dart --seed 0 --ray_beta 0.8      # extra args go to tests/main.py

Methods:
  metra          METRA (Park et al., ICLR 2024)
  dart           DART, constant beta (--reward_mode ray): SAC reward z^T dphi - beta [b(u_{t+1}) - b(u_t)],
                 u_t = phi(s_t) - phi(s_0), b(u) = |u - (z^T u) z| (lateral deviation from the skill ray); the policy
                 and Q functions see (s, u_t, z); phi's objective and constraint are METRA's. beta = 0.5; the penalty
                 is off for the first 10% of epochs and ramps linearly to its full value at 20% (--ray_warmup).
  dart_adaptive  DART, direction-adaptive beta(psi) = beta_base (1 + e(psi))^eta / mean_psi (1 + e)^eta, updated every
                 epoch from the end-angle errors e = (1 - cos angle(u_T, z)) / 2 of that epoch's training trajectories
                 (3600 angular bins, eta = 1, beta_base = 0.5)
  nm             norm matching (Atanassov et al., 2024): per-step target dphi = nm_step z, skill radius R = 200 nm_step = 24
  d3             D3 schedule (Cathomen et al., CoRL 2025): METRA, switching to norm matching as cos(dphi, z) rises
                 from 0.5 to 0.7

Each run trains for 10001 epochs (8 trajectories x 200 steps per epoch, 1.6e7 env steps) and writes checkpoints and
logs to exp/<env>_<method>/sd<seed>_<timestamp>_.../ (option_policy10000.pt and traj_encoder10000.pt are used by eval/).
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
COMMON = [
    '--max_path_length', '200',
    '--traj_batch_size', '8',
    '--n_parallel', '1',
    '--normalizer_type', 'preset',
    '--trans_optimization_epochs', '50',
    '--n_epochs_per_log', '100',
    '--n_epochs_per_eval', '1000',
    '--n_epochs_per_save', '10000',
    '--sac_max_buffer_size', '1000000',
    '--algo', 'metra',
    '--discrete', '0',
    '--dim_option', '2',
    '--n_epochs', '10001',
]
ENV_ARGS = {
    'ant': ['--env', 'ant', '--eval_plot_axis', '-50', '50', '-50', '50'],
    'half_cheetah': ['--env', 'half_cheetah'],
}
NM_STEP = 0.120  # R = 200 * NM_STEP = 24 (see README)
METHOD_ARGS = {
    'metra': [],
    'dart': ['--reward_mode', 'ray', '--ray_beta', '0.5'],
    'dart_adaptive': ['--reward_mode', 'ray', '--ray_beta', '0.5', '--ray_beta_mode', 'adaptive'],
    'nm': ['--reward_mode', 'norm_match', '--nm_step', str(NM_STEP), '--nm_sigma', '10',
           '--dual_dist', 'l2', '--dual_lam', '30', '--dual_slack', '1e-6'],
    'd3': ['--reward_mode', 'd3', '--nm_sigma', '10', '--d3_switch_range', '0.5', '0.7'],
}


def mujoco_env():
    """mujoco-py needs MuJoCo 2.1's bin directory on LD_LIBRARY_PATH."""
    env = dict(os.environ)
    paths = [p for p in env.get('LD_LIBRARY_PATH', '').split(':') if p]
    for p in [os.path.expanduser('~/.mujoco/mujoco210/bin'), '/usr/lib/nvidia']:
        if p not in paths:
            paths.append(p)
    env['LD_LIBRARY_PATH'] = ':'.join(paths)
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', required=True, choices=list(ENV_ARGS))
    ap.add_argument('--method', required=True, choices=list(METHOD_ARGS))
    ap.add_argument('--seed', type=int, default=0)
    args, extra = ap.parse_known_args()
    cmd = [sys.executable, 'tests/main.py', *COMMON, *ENV_ARGS[args.env], *METHOD_ARGS[args.method],
           '--run_group', f'{args.env}_{args.method}', '--seed', str(args.seed), *extra]
    print(' '.join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=REPO, env=mujoco_env()))


if __name__ == '__main__':
    main()
