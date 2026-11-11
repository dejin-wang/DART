#!/usr/bin/env python3
"""Random-start robustness of the skill directions (Ant, frozen policies, no training).

Same rollout as rollout_2000.py (deterministic policy, 200 steps, evenly spaced z) but from other start states:
  fixed  - the fixed start of rollout_2000.py (np seed 0 reset), as the reference at the same skill count
  static - K starts: reset noise with np seed 100 + k (qpos +-0.1, qvel ~ 0.1 N(0, 1)), torso yawed by 45 k deg (wrapped to [-180, 180), so quaternion w >= 0), at rest
  moving - as static, then a 50-step pre-roll with skill z at the golden angle 137.5 k deg, so the ant starts mid-gait
           with its own velocity; the skill (u_t for DART) restarts at the end of the pre-roll
All N skills of a start share one exact start state (qpos, qvel copied into every env).

    python eval/ant/rollout_randstart.py --run <run dir> --label <label>
Writes results/ant/randstart/<label>.npz: per condition c, c_z (deg), c_xy (S, N, T/10+1, 2) world x-y
relative to the start, c_u (S, N, 2) = phi(s_T) - phi(s_0), c_yaw (S,) start yaw (deg), c_up (S, N) final world z of the torso up axis.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'results', 'ant', 'randstart')
sys.path[:0] = [REPO, os.path.join(REPO, 'garaged', 'src')]
os.chdir(REPO)
torch.set_num_threads(2)

from envs.mujoco.ant_env import AntEnv, q_mult
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--epoch', type=int, default=10000)
ap.add_argument('--n', type=int, default=500, help='evenly spaced skills per start')
ap.add_argument('--starts', type=int, default=8)
ap.add_argument('--pre', type=int, default=50, help='pre-roll steps of the moving starts')
args = ap.parse_args()

policy = torch.load(os.path.join(args.run, f'option_policy{args.epoch}.pt'), map_location='cpu')['policy']
policy._force_use_mode_actions = True
phi_net = torch.load(os.path.join(args.run, f'traj_encoder{args.epoch}.pt'), map_location='cpu')['traj_encoder']
MEAN, STD = get_normalizer_preset('ant_preset')
T = 200
z_deg = np.arange(args.n) * 360 / args.n
Z = np.stack([np.cos(np.radians(z_deg)), np.sin(np.radians(z_deg))], 1)
make = lambda: consistent_normalize(AntEnv(render_hw=100), normalize_obs=True, mean=MEAN, std=STD)
envs = [make() for _ in range(args.n)]


def phi(o):
    with torch.no_grad():
        return phi_net(torch.as_tensor(o, dtype=torch.float32)).mean.numpy()


def start_state(k, moving):
    """(qpos, qvel) of start k."""
    e = make()
    np.random.seed(100 + k)
    e.reset()
    sim = e.unwrapped.sim
    qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    yaw = np.radians((45 * k + 180) % 360 - 180)  # in [-180, 180): quaternion w >= 0, as MuJoCo produces when turning
    qpos[3:7] = q_mult(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), qpos[3:7])
    qvel[:2] = [np.cos(yaw) * qvel[0] - np.sin(yaw) * qvel[1], np.sin(yaw) * qvel[0] + np.cos(yaw) * qvel[1]]
    e.unwrapped.set_state(qpos, qvel)
    if moving:
        e._cur_obs = e.unwrapped._get_obs()
        obs = e._apply_normalize_obs(e._cur_obs)[None]
        a = np.radians(137.5 * k)
        zp = np.array([[np.cos(a), np.sin(a)]])
        policy.reset()
        for _ in range(args.pre):
            act, _ = policy.get_actions(np.concatenate([obs, zp], 1).astype(np.float32))
            obs = e.step(act[0])[0][None]
    return sim.data.qpos.copy(), sim.data.qvel.copy(), np.degrees(yaw)


def rollout(qpos, qvel):
    obs = []
    for e in envs:
        e.reset()
        e.unwrapped.set_state(qpos, qvel)
        e._cur_obs = e.unwrapped._get_obs()
        obs.append(e._apply_normalize_obs(e._cur_obs))
    obs = np.stack(obs)
    xy0 = qpos[:2].copy()
    phi0 = phi(obs)
    policy.reset()
    xy = [np.zeros((args.n, 2))]
    for t in range(T):
        act, _ = policy.get_actions(np.concatenate([obs, Z], 1).astype(np.float32))
        obs = np.stack([e.step(x)[0] for e, x in zip(envs, act)])
        if (t + 1) % 10 == 0:
            xy.append(np.stack([e.unwrapped.sim.data.qpos.flat[:2].copy() for e in envs]) - xy0)
    q = np.stack([e.unwrapped.sim.data.qpos[3:7].copy() for e in envs])
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # world z of the torso's up axis (< 0: flipped)
    return np.stack(xy, 1), phi(obs) - phi0, up


res = {}
for cond in ['fixed', 'static', 'moving']:
    XY, U, YAW, UP = [], [], [], []
    for k in range(1 if cond == 'fixed' else args.starts):
        if cond == 'fixed':
            e = make()
            np.random.seed(0)
            e.reset()
            qpos, qvel, yaw = e.unwrapped.sim.data.qpos.copy(), e.unwrapped.sim.data.qvel.copy(), 0.
        else:
            qpos, qvel, yaw = start_state(k, cond == 'moving')
        xy, u, up = rollout(qpos, qvel)
        UP.append(up)
        XY.append(xy)
        U.append(u)
        YAW.append(yaw)
        print(args.label, cond, k, 'mean dist', np.linalg.norm(xy[:, -1], axis=1).mean().round(1), flush=True)
    res[f'{cond}_z'] = z_deg
    res[f'{cond}_xy'] = np.stack(XY).astype(np.float32)
    res[f'{cond}_u'] = np.stack(U).astype(np.float32)
    res[f'{cond}_yaw'] = np.array(YAW)
    res[f'{cond}_up'] = np.stack(UP).astype(np.float32)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(f'{OUT}/{args.label}.npz', **res)
print('saved', f'{OUT}/{args.label}.npz')
