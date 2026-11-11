#!/usr/bin/env python3
"""HalfCheetah evaluation of frozen skills with one protocol for every method (no training).

Exploration: N evenly spaced unit z, deterministic policy, 200 steps (METRA's HalfCheetah episode length) from the fixed
start (np seed 0 reset). Saves full qpos trajectories (x, rootz, pitch, 6 joints) and u_T = phi(s_T) - phi(s_0).

HalfCheetahGoal (METRA App. F.1): goal x uniform in [-100, 100] (np RandomState(0), the same G goals for every method),
success = |x - g| <= 3, episode 200 steps, start = env reset with np seed 1000 + i for goal i. Zero-shot controller of
METRA Sec. 4.2 / App. F.2: z = normalize(phi(g) - phi(s)) re-computed every step; goal observation = current
observation with x replaced by g. Ray (DART) policies: u_t restarts at the episode start (policy.reset()).

    python eval/half_cheetah/hc_protocol_eval.py --run <run dir> --label <label>
        -> results/half_cheetah/protocol/<label>.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'results', 'half_cheetah', 'protocol')
sys.path[:0] = [REPO, os.path.join(REPO, 'garaged', 'src')]
os.chdir(REPO)
torch.set_num_threads(2)

from envs.mujoco.half_cheetah_env import HalfCheetahEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--epoch', type=int, default=10000)
ap.add_argument('--n', type=int, default=360, help='evenly spaced skills (exploration)')
ap.add_argument('--goals', type=int, default=256)
args = ap.parse_args()
T, TOL, GRANGE = 200, 3.0, 100.0
MEAN, STD = get_normalizer_preset('half_cheetah_preset')

policy = torch.load(f'{args.run}/option_policy{args.epoch}.pt', map_location='cpu')['policy']
policy._force_use_mode_actions = True
phi_net = torch.load(f'{args.run}/traj_encoder{args.epoch}.pt', map_location='cpu')['traj_encoder']
make = lambda: consistent_normalize(HalfCheetahEnv(render_hw=100), normalize_obs=True, mean=MEAN, std=STD)


def phi(o):
    with torch.no_grad():
        return phi_net(torch.as_tensor(o, dtype=torch.float32)).mean.numpy()


def unit(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def reset(seeds):
    envs, obs = [], []
    for s in seeds:
        e = make()
        np.random.seed(s)
        obs.append(e.reset())
        envs.append(e)
    return envs, np.stack(obs)


def act(obs, z):
    return policy.get_actions(np.concatenate([obs, z], 1).astype(np.float32))[0]


qpos = lambda envs: np.stack([e.unwrapped.sim.data.qpos.flat.copy() for e in envs])
res = {}

# Exploration
z_deg = np.arange(args.n) * 360 / args.n
Z = np.stack([np.cos(np.radians(z_deg)), np.sin(np.radians(z_deg))], 1)
envs, obs = reset([0] * args.n)
phi0 = phi(obs)
policy.reset()
Q = [qpos(envs)]
for t in range(T):
    obs = np.stack([e.step(a)[0] for e, a in zip(envs, act(obs, Z))])
    Q.append(qpos(envs))
res.update(z=z_deg, qpos=np.stack(Q, 1).astype(np.float32), u=(phi(obs) - phi0).astype(np.float32))
print(args.label, 'exploration: x_T range', np.ptp(res['qpos'][:, -1, 0] - res['qpos'][:, 0, 0]).round(1), flush=True)

# HalfCheetahGoal
G = args.goals
envs, obs = reset([1000 + i for i in range(G)])
x0 = qpos(envs)[:, 0]
goals = np.random.RandomState(0).uniform(-GRANGE, GRANGE, G)
policy.reset()
X = [x0.copy()]
for t in range(T):
    raw = np.stack([e._cur_obs.copy() for e in envs])
    raw[:, 0] = goals
    z = unit(phi((raw - MEAN) / (STD + 1e-8)) - phi(obs))
    obs = np.stack([e.step(a)[0] for e, a in zip(envs, act(obs, z))])
    X.append(qpos(envs)[:, 0])
res.update(goal=goals, goal_x=np.stack(X, 1).astype(np.float32))
d = np.abs(res['goal_x'] - goals[:, None])
print(args.label, 'goal: any', (d <= TOL).any(1).mean().round(3), 'final', (d[:, -1] <= TOL).mean().round(3),
      'final dist', d[:, -1].mean().round(2), flush=True)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(f'{OUT}/{args.label}.npz', **res)
print('saved', f'{OUT}/{args.label}.npz')
