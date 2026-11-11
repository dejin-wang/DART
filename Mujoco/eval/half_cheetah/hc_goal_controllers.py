#!/usr/bin/env python3
"""HalfCheetahGoal with frozen skills: training-free controllers and a high-level SAC (METRA App. F.1/F.2).

Task (as hc_protocol_eval.py): goal x ~ U[-100, 100], success |x - g| <= 3 (reward 10, episode ends), 200 steps.
Evaluation set: the same 256 goals (np RandomState(0)) and starts (reset with np seed 1000 + i) as hc_protocol_eval.py.
d = normalize(phi(goal obs) - phi(s)), goal obs = current obs with x replaced by g.
  --mode free: h1 (z = d every step; = hc_protocol_eval.py), h25 (every 25 steps), adr (ADR as Ant goal_reach.py:
               h_k = clip(floor((|x - g| - 3)_+ / (2 v_k + 0.01)), 0, 25), v_k = |dx| / steps of the previous segment,
               first segment h_max; h_k = 0 -> zero action, rechecked every step), adr_nostop (h_k = max(1, ...)).
               u_t (DART) restarts at the episode start only.
  --mode sac:  high-level SAC picks a unit z every K = 25 steps (eval/ant/ant_multigoals_hrl.py SAC, same hyper-parameters,
               u_t restarts each segment); obs = normalised state, (g - x) / 100, t / 200; training goals/starts from
               --seed; 8000 training episodes (1.6M env steps); evaluated on the 256-goal set every 800 episodes.
Success metrics on the evaluation set: any-step (= return / 10), final-step (the agent keeps acting after success; for
SAC the episode would end, so final-step is reported for the training-free controllers only), final |x - g|.

    python eval/half_cheetah/hc_goal_controllers.py --run <run> --label <label> --mode free|sac
        -> results/half_cheetah/goal_ctrl/<label>_<mode>.npz (+ .csv learning curve for sac)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'results', 'half_cheetah', 'goal_ctrl')
sys.path[:0] = [REPO, os.path.join(REPO, 'garaged', 'src'), os.path.join(REPO, 'eval', 'ant')]
os.chdir(REPO)
torch.set_num_threads(2)

from envs.mujoco.half_cheetah_env import HalfCheetahEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset
from ant_multigoals_hrl import SAC

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--mode', required=True, choices=['free', 'sac'])
ap.add_argument('--epoch', type=int, default=10000)
ap.add_argument('--goals', type=int, default=256)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n_envs', type=int, default=16)
ap.add_argument('--episodes', type=int, default=8000)
ap.add_argument('--eval_every', type=int, default=800)
args = ap.parse_args()
T, K, TOL, GRANGE, REWARD, ADR_EPS = 200, 25, 3.0, 100.0, 10.0, 0.01
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


def act(obs, z):
    return policy.get_actions(np.concatenate([obs, z], 1).astype(np.float32))[0]


class Goal:
    """n HalfCheetah envs in lockstep."""

    def __init__(self, n):
        self.envs = [make() for _ in range(n)]
        self.n = n

    def reset(self, seeds, goals):
        obs = []
        for e, s in zip(self.envs, seeds):
            if s is not None:
                np.random.seed(s)
            obs.append(e.reset())
        self.obs, self.t = np.stack(obs), 0
        self.goal = goals.astype(np.float64)
        self.done = np.zeros(self.n, bool)
        return self.obs

    def x(self):
        return np.array([e.unwrapped.sim.data.qpos[0] for e in self.envs])

    def d(self):
        raw = np.stack([e._cur_obs.copy() for e in self.envs])
        raw[:, 0] = self.goal
        return unit(phi((raw - MEAN) / (STD + 1e-8)) - phi(self.obs))

    def step(self, a):
        self.obs = np.stack([e.step(x)[0] for e, x in zip(self.envs, a)])
        self.t += 1
        hit = ~self.done & (np.abs(self.x() - self.goal) <= TOL)
        self.done |= hit
        return hit

    def hl_obs(self):
        return np.concatenate([self.obs, ((self.goal - self.x()) / GRANGE)[:, None],
                               np.full((self.n, 1), self.t / T)], 1).astype(np.float32)


EVAL_SEEDS = [1000 + i for i in range(args.goals)]
EVAL_GOALS = np.random.RandomState(0).uniform(-GRANGE, GRANGE, args.goals)
os.makedirs(OUT, exist_ok=True)
res = {'goal': EVAL_GOALS}

if args.mode == 'free':
    env = Goal(args.goals)
    for ctrl in ['h1', 'h25', 'adr', 'adr_nostop']:
        env.reset(EVAL_SEEDS, EVAL_GOALS)
        policy.reset()
        z, countdown, sel = np.zeros((env.n, 2)), np.zeros(env.n, int), np.zeros(env.n, int)
        vhat, seg_x0, seg_h, freeze = np.full(env.n, np.nan), np.zeros(env.n), np.zeros(env.n, int), np.zeros(env.n, bool)
        X = [env.x()]
        for t in range(T):
            d = env.d()
            if ctrl == 'h1':
                due = np.ones(env.n, bool)
                sel += 1
            elif ctrl == 'h25':
                due = np.full(env.n, t % K == 0)
                sel += due
            else:
                due = countdown <= 0
                if due.any():
                    cur = X[-1]
                    moved = due & (seg_h > 0) & ~freeze
                    vhat[moved] = np.abs(cur[moved] - seg_x0[moved]) / seg_h[moved]
                    h = np.floor(np.maximum(np.abs(env.goal[due] - cur[due]) - TOL, 0) / (2 * np.nan_to_num(vhat[due]) + ADR_EPS))
                    h = np.where(np.isnan(vhat[due]), K, np.clip(h, 0, K)).astype(int)
                    if ctrl == 'adr_nostop':
                        h = np.maximum(h, 1)
                    s = np.zeros(env.n, bool)
                    s[due] = h > 0
                    sel += s
                    freeze[due] = h == 0
                    seg_x0[due], seg_h[due] = cur[due], h
                    countdown[due] = np.where(h == 0, 1, h)
                countdown -= 1
            z = np.where(due[:, None], d, z)
            a = act(env.obs, z)
            if ctrl == 'adr':
                a = np.where(freeze[:, None], 0.0, a)
            env.step(a)
            X.append(env.x())
        X = np.stack(X, 1)
        res[f'{ctrl}_x'], res[f'{ctrl}_sel'] = X.astype(np.float32), sel
        dist = np.abs(X - EVAL_GOALS[:, None])
        print(args.label, ctrl, 'any', (dist <= TOL).any(1).mean().round(3), 'final', (dist[:, -1] <= TOL).mean().round(3),
              'final dist', dist[:, -1].mean().round(2), 'sel', sel.mean().round(1), flush=True)
    np.savez_compressed(f'{OUT}/{args.label}_free.npz', **res)
    sys.exit(0)

# high-level SAC
torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
np.random.seed(args.seed)


def hl_episode(env, choose, store=None):
    """One episode: z every K steps (u_t restarts each segment); returns per-env return and x trajectory."""
    o = env.hl_obs()
    ret, X = np.zeros(env.n), [env.x()]
    while env.t < T:
        active = ~env.done
        a = choose(o)
        z = unit(a)
        policy.reset()
        r = np.zeros(env.n)
        for _ in range(K):
            r += REWARD * env.step(act(env.obs, z))
            X.append(env.x())
        o2 = env.hl_obs()
        ret += r
        if store is not None:
            store(o[active], a[active], r[active], o2[active], (env.done | (env.t >= T))[active].astype(np.float32))
        o = o2
    return ret, np.stack(X, 1)


eval_env = Goal(args.goals)


def evaluate():
    eval_env.reset(EVAL_SEEDS, EVAL_GOALS)
    ret, X = hl_episode(eval_env, lambda o: sac.act(o, True))
    return ret, X


env = Goal(args.n_envs)
obs_dim = env.reset([None] * env.n, np.zeros(env.n)).shape[1] + 2
sac = SAC(obs_dim)
cap = 1_000_000
buf = {k: np.zeros((cap,) + s, np.float32) for k, s in [('o', (obs_dim,)), ('a', (2,)), ('r', ()), ('o2', (obs_dim,)), ('d', ())]}
size = 0


def store(o, a, r, o2, d):
    global size
    idx = np.arange(size, size + len(o)) % cap
    for k, v in [('o', o), ('a', a), ('r', r), ('o2', o2), ('d', d)]:
        buf[k][idx] = v
    size += len(o)


def choose_train(o):
    if size < 2000:
        return rng.uniform(-1, 1, (len(o), 2)).astype(np.float32)
    return sac.act(o)


rows, episodes, start = [], 0, time.time()
ret, _ = evaluate()
rows.append(dict(episodes=0, env_steps=0, success=ret.mean() / REWARD))
while episodes < args.episodes:
    env.reset([None] * env.n, rng.uniform(-GRANGE, GRANGE, env.n))
    n0 = size
    hl_episode(env, choose_train, store)
    if size >= 2000:
        for _ in range(size - n0):
            j = np.random.randint(0, min(size, cap), 256)
            sac.update([torch.as_tensor(buf[k][j]) for k in ['o', 'a', 'r', 'o2', 'd']])
    episodes += env.n
    if episodes % args.eval_every < env.n:
        ret, X = evaluate()
        rows.append(dict(episodes=episodes, env_steps=episodes * T, success=ret.mean() / REWARD))
        print(f'{args.label}: {episodes} episodes, eval success {ret.mean() / REWARD:.3f} '
              f'[{(time.time() - start) / 60:.0f} min]', flush=True)
with open(f'{OUT}/{args.label}_sac.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['episodes', 'env_steps', 'success'])
    w.writeheader()
    w.writerows(rows)
res.update(sac_x=X.astype(np.float32), sac_ret=ret, sac_curve=np.array([r['success'] for r in rows]))
np.savez_compressed(f'{OUT}/{args.label}_sac.npz', **res)
print(f'{args.label}: FINAL SAC success (mean of last 3 evals) {np.mean([r["success"] for r in rows[-3:]]):.3f}', flush=True)
