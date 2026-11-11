#!/usr/bin/env python3
"""AntMultiGoals hierarchical control with frozen skills (METRA paper, App. F).

Task: reach four goals in turn (radius 3 m), each sampled uniformly within +-7.5 m of the ant's current x-y;
+2.5 per goal reached; a new goal is drawn when the current one is reached or after 50 steps; episodes last
200 steps (maximum return 10). A high-level policy picks a unit skill z every K = 25 steps and the frozen skill
policy acts for those 25 steps (the ray policy measures u_t from the start of each 25-step segment).

The high-level policy is trained with SAC (continuous z, as in the paper). High-level observation: normalised
state, goal offset / 7.5, steps spent on the current goal / 50, goals done / 4, time / 200. A training-free
reference controller, z = normalize(phi(goal state) - phi(s)) at every high-level step, is also evaluated.

Norm-matching runs (norm_match / d3), whose skill length also sets the speed: --skill_space disk lets the high-level
action keep its length (clipped to the unit disk) instead of being normalised; with --nm_step a second reference,
z = (phi(goal) - phi(s)) / (K * nm_step) clipped to the disk (reach the goal within the segment), is also evaluated.

    python eval/ant/ant_multigoals_hrl.py --run <run dir> --label metra_s0
Writes <out>/<label>.csv (default results/ant/multigoals_sac/) (evaluation return over training) and prints the final return.
"""
import argparse
import csv
import os
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO_DIR, os.path.join(REPO_DIR, 'garaged', 'src')]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.mujoco.ant_env import AntEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

K, EP_LEN, GOAL_STEPS, N_GOALS, RADIUS, GOAL_RANGE, GOAL_REWARD = 25, 200, 50, 4, 3.0, 7.5, 2.5


class MultiGoals:
    """N Ant envs stepped in lockstep; one high-level step = K low-level steps of the frozen skill policy."""

    def __init__(self, n, skill_policy, phi_net, seed, disk=False):
        mean, std = get_normalizer_preset('ant_preset')
        self.mean, self.std = mean, std
        self.envs = [consistent_normalize(AntEnv(render_hw=100), normalize_obs=True, mean=mean, std=std) for _ in range(n)]
        self.policy, self.phi = skill_policy, phi_net
        self.rng = np.random.RandomState(seed)
        self.n = n
        self.disk = disk  # True: skills keep their length (clipped to the unit disk); False: unit skills

    def xy(self):
        return np.stack([e.unwrapped.sim.data.qpos.flat[:2].copy() for e in self.envs])

    def hl_obs(self):
        return np.concatenate([self.obs, (self.goal - self.xy()) / GOAL_RANGE, self.goal_t[:, None] / GOAL_STEPS,
                               self.goals_done[:, None] / N_GOALS, np.full((self.n, 1), self.t / EP_LEN)], 1).astype(np.float32)

    def reset(self):
        self.obs = np.stack([e.reset() for e in self.envs])
        self.t = 0
        self.goal = self.xy() + self.rng.uniform(-GOAL_RANGE, GOAL_RANGE, (self.n, 2))
        self.goal_t = np.zeros(self.n)
        self.goals_done = np.zeros(self.n)
        return self.hl_obs()

    def reference_z(self, nm_step=None):
        raw = np.stack([e._cur_obs.copy() for e in self.envs])
        raw[:, :2] = self.goal
        with torch.no_grad():
            d = (self.phi(torch.as_tensor((raw - self.mean) / (self.std + 1e-8), dtype=torch.float32)).mean
                 - self.phi(torch.as_tensor(self.obs, dtype=torch.float32)).mean)
        if nm_step is not None:
            z = d / (K * nm_step)
            return (z / z.norm(dim=1, keepdim=True).clamp(min=1)).numpy()
        return (d / (d.norm(dim=1, keepdim=True) + 1e-8)).numpy()

    def step(self, z):
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / np.maximum(norm, 1) if self.disk else z / (norm + 1e-8)
        reward = np.zeros(self.n)
        self.policy.reset()
        for _ in range(K):
            actions, _ = self.policy.get_actions(np.concatenate([self.obs, z], 1))
            self.obs = np.stack([e.step(a)[0] for e, a in zip(self.envs, actions)])
            self.t += 1
            active = self.goals_done < N_GOALS
            self.goal_t += active
            reached = active & (np.linalg.norm(self.xy() - self.goal, axis=1) <= RADIUS)
            timeout = active & ~reached & (self.goal_t >= GOAL_STEPS)
            reward += GOAL_REWARD * reached
            new = reached | timeout
            self.goals_done += new
            self.goal[new] = self.xy()[new] + self.rng.uniform(-GOAL_RANGE, GOAL_RANGE, (new.sum(), 2))
            self.goal_t[new] = 0
        return self.hl_obs(), reward.astype(np.float32), self.t >= EP_LEN


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class SAC:
    def __init__(self, obs_dim, act_dim=2, lr=3e-4, gamma=0.99, tau=5e-3):
        self.actor = mlp(obs_dim, 2 * act_dim)
        self.q = nn.ModuleList([mlp(obs_dim + act_dim, 1) for _ in range(2)])
        self.q_targ = nn.ModuleList([mlp(obs_dim + act_dim, 1) for _ in range(2)])
        self.q_targ.load_state_dict(self.q.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr)
        self.gamma, self.tau, self.target_entropy = gamma, tau, -act_dim

    def dist(self, o):
        mu, log_std = self.actor(o).chunk(2, -1)
        return mu, log_std.clamp(-5, 2).exp()

    def sample(self, o):
        mu, std = self.dist(o)
        x = mu + std * torch.randn_like(mu)
        a = torch.tanh(x)
        logp = (torch.distributions.Normal(mu, std).log_prob(x) - torch.log(1 - a ** 2 + 1e-6)).sum(-1)
        return a, logp

    @torch.no_grad()
    def act(self, o, deterministic=False):
        o = torch.as_tensor(o)
        return (torch.tanh(self.dist(o)[0]) if deterministic else self.sample(o)[0]).numpy()

    def update(self, b):
        o, a, r, o2, d = b
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.sample(o2)
            qt = torch.min(*[q(torch.cat([o2, a2], -1)).squeeze(-1) for q in self.q_targ])
            y = r + self.gamma * (1 - d) * (qt - alpha * logp2)
        loss_q = sum(F.mse_loss(q(torch.cat([o, a], -1)).squeeze(-1), y) for q in self.q)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        a_new, logp = self.sample(o)
        loss_a = (alpha * logp - torch.min(*[q(torch.cat([o, a_new], -1)).squeeze(-1) for q in self.q])).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)


def evaluate(env, choose, n_episodes):
    returns = []
    while len(returns) < n_episodes:
        o, ret, done = env.reset(), np.zeros(env.n), False
        while not done:
            o, r, done = env.step(choose(o))
            ret += r
        returns.extend(ret.tolist())
    return float(np.mean(returns[:n_episodes]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', default=os.path.join(REPO_DIR, 'results', 'ant', 'multigoals_sac'))
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=8000, help='training episodes (x200 env steps)')
    parser.add_argument('--eval_every', type=int, default=800)
    parser.add_argument('--eval_episodes', type=int, default=96)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skill_space', default='unit', choices=['unit', 'disk'])
    parser.add_argument('--nm_step', type=float, default=None)
    args = parser.parse_args()

    os.chdir(REPO_DIR)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    skill = torch.load(os.path.join(args.run, f'option_policy{args.epoch}.pt'), map_location='cpu')['policy']
    skill._force_use_mode_actions = True
    phi = torch.load(os.path.join(args.run, f'traj_encoder{args.epoch}.pt'), map_location='cpu')['traj_encoder']
    disk = args.skill_space == 'disk'
    env = MultiGoals(args.n_envs, skill, phi, seed=args.seed, disk=disk)
    eval_env = MultiGoals(args.n_envs, skill, phi, seed=10_000 + args.seed, disk=disk)

    eval_env.disk = False
    ref = evaluate(eval_env, lambda o: eval_env.reference_z(), args.eval_episodes)
    print(f'{args.label}: training-free reference controller (z from phi(goal)) return {ref:.2f}', flush=True)
    ref_native = None
    if args.nm_step is not None:
        eval_env.disk = True
        ref_native = evaluate(eval_env, lambda o: eval_env.reference_z(args.nm_step), args.eval_episodes)
        print(f'{args.label}: training-free reference, z = (phi(goal) - phi(s)) / (K nm_step) in the disk, '
              f'return {ref_native:.2f}', flush=True)
    eval_env.disk = disk

    obs_dim = env.reset().shape[1]
    sac = SAC(obs_dim)
    cap = 1_000_000
    buf = {k: np.zeros((cap,) + s, np.float32) for k, s in
           [('o', (obs_dim,)), ('a', (2,)), ('r', ()), ('o2', (obs_dim,)), ('d', ())]}
    size, episodes, rows, start = 0, 0, [], time.time()
    rows.append({'episodes': 0, 'env_steps': 0, 'return': evaluate(eval_env, lambda o: sac.act(o, True), args.eval_episodes)})
    while episodes < args.episodes:
        o, done = env.reset(), False
        while not done:
            a = np.random.uniform(-1, 1, (env.n, 2)).astype(np.float32) if size < 2000 else sac.act(o)
            o2, r, done = env.step(a)
            idx = np.arange(size, size + env.n) % cap
            for k, v in [('o', o), ('a', a), ('r', r), ('o2', o2), ('d', np.full(env.n, float(done)))]:
                buf[k][idx] = v
            size += env.n
            o = o2
            if size >= 2000:
                for _ in range(env.n):
                    j = np.random.randint(0, min(size, cap), 256)
                    sac.update([torch.as_tensor(buf[k][j]) for k in ['o', 'a', 'r', 'o2', 'd']])
        episodes += env.n
        if episodes % args.eval_every < env.n:
            ret = evaluate(eval_env, lambda o: sac.act(o, True), args.eval_episodes)
            rows.append({'episodes': episodes, 'env_steps': episodes * EP_LEN, 'return': ret})
            print(f'{args.label}: {episodes} episodes ({episodes * EP_LEN / 1e6:.2f}M steps) return {ret:.2f} '
                  f'[{(time.time() - start) / 60:.0f} min]', flush=True)

    with open(os.path.join(args.out, f'{args.label}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, ['episodes', 'env_steps', 'return', 'reference_return']
                           + (['reference_native_return'] if ref_native is not None else []))
        w.writeheader()
        for row in rows:
            w.writerow({**row, 'reference_return': ref,
                        **({'reference_native_return': ref_native} if ref_native is not None else {})})
    final = np.mean([r['return'] for r in rows[-3:]])
    print(f'{args.label}: FINAL high-level SAC return (mean of last 3 evals) {final:.2f} | reference {ref:.2f}', flush=True)


if __name__ == '__main__':
    main()
