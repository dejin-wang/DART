#!/usr/bin/env python3
"""Fixed-start rollouts of 2000 evenly spaced skills for one run (np seed 0 before each reset, deterministic policy,
200 steps), recording qpos[:15] and phi(s_t) of every step.

    python eval/ant/rollout_2000.py --run <run dir> --label <label>
Writes results/ant/rollouts/<label>.npz (read by eval/ant/summarize.py).
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'results', 'ant', 'rollouts')
sys.path[:0] = [REPO, os.path.join(REPO, 'garaged', 'src')]
os.chdir(REPO)
torch.set_num_threads(2)

from envs.mujoco.ant_env import AntEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--n', type=int, default=2000)
parser.add_argument('--chunk', type=int, default=500, help='environments stepped together')
args = parser.parse_args()

policy = torch.load(os.path.join(args.run, f'option_policy{args.epoch}.pt'), map_location='cpu')['policy']
policy._force_use_mode_actions = True
phi_net = torch.load(os.path.join(args.run, f'traj_encoder{args.epoch}.pt'), map_location='cpu')['traj_encoder']
mean, std = get_normalizer_preset('ant_preset')

z_deg = np.arange(args.n) * 360 / args.n
Q, P = [], []
for i in range(0, args.n, args.chunk):
    zc = np.stack([np.cos(np.radians(z_deg[i:i + args.chunk])), np.sin(np.radians(z_deg[i:i + args.chunk]))], 1)
    envs = [consistent_normalize(AntEnv(render_hw=100), normalize_obs=True, mean=mean, std=std) for _ in zc]
    obs = []
    for e in envs:
        np.random.seed(0)  # the same start state for every rollout
        obs.append(e.reset())
    obs = np.stack(obs)
    policy.reset()
    O = [obs]
    q = lambda: np.stack([e.unwrapped.sim.data.qpos.flat[:15].copy() for e in envs])
    Qc = [q()]
    for _ in range(200):
        act, _ = policy.get_actions(np.concatenate([obs, zc], 1))
        obs = np.stack([e.step(x)[0] for e, x in zip(envs, act)])
        O.append(obs)
        Qc.append(q())
    O = np.stack(O, 1)
    with torch.no_grad():
        P.append(phi_net(torch.as_tensor(O.reshape(-1, O.shape[-1]), dtype=torch.float32)).mean.numpy()
                 .reshape(len(zc), O.shape[1], -1))
    Q.append(np.stack(Qc, 1))
os.makedirs(OUT, exist_ok=True)
f = f'{OUT}/{args.label}.npz'
np.savez(f, z=z_deg, qpos=np.concatenate(Q).astype(np.float32), phi=np.concatenate(P).astype(np.float32),
         run=os.path.abspath(args.run))
print('saved', f)
