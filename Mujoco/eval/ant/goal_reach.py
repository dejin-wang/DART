#!/usr/bin/env python3
"""Ant zero-shot goal reaching with frozen skills (no training): moving-start goals and AntMultiGoals.

Moving start: every episode starts from the same reset state (np seed 0 before each reset); the ant first walks for
50 steps towards +x (goal-bearing skill to a point 20 m ahead, two 25-step segments), so that it has a current motion
direction t. Then N goals are placed around it in its motion frame: angle a_i evenly spaced in (-180, 180] deg from t,
radius 10-40 m (golden-ratio sequence). Goals with |a| > 120 deg are "behind". Goal phase: B steps (default 400),
success tolerance 3 m. The skill is d = normalize(phi(g) - phi(s)), g = current state with x-y replaced by the goal.
Controllers:
  bearing1       - z = d every step (fixed h = 1)
  bearing25      - z = d every 25 steps (fixed h = 25)
  adr            - Distance-Adaptive Replanning: z = d_k, held for
                   h_k = max(0, min(h_max, floor((D_k - r_g)_+ / (2 v_k + eps)))) steps, v_k = x-y net displacement /
                   steps of the previous segment (seeded from the pre-roll); h_k = 0 (inside the success ball) gives a
                   zero action and is rechecked every step (h_max = 25, r_g = 3 m, eps = 0.01)
Stop / no-stop ablation ('stop' = zero action while the x-y goal distance is <= 3 m):
  bearing1_stop  - bearing1 + stop
  bearing25_stop - bearing25 + stop; per-env 25-step schedule that replans as soon as the ant leaves the ball
  adr_nostop     - adr with h_k = max(1, ...), never frozen
Every controller records `selections` = high-level skill choices made while not stopped.
DART policies measure u_t = phi(s_t) - phi(s_start) from the goal-phase start.

AntMultiGoals (METRA App. F; MultiGoals in ant_multigoals_hrl.py, eval seed 10000): four goals in turn, each within
+-7.5 m of the ant, radius 3 m, 50 steps per goal, 200 steps, +2.5 per goal reached (max 10). Controller: ADR on its
own per-env schedule (D_k = distance to the current goal), u_t restarting every 25 steps.

    python eval/ant/goal_reach.py --run <run dir> --label <label>      -> results/ant/goal_reach/<label>.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, 'garaged', 'src'), os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
torch.set_num_threads(2)

from envs.mujoco.ant_env import AntEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset
from ant_multigoals_hrl import EP_LEN, GOAL_RANGE, GOAL_REWARD, GOAL_STEPS, K, N_GOALS, RADIUS, MultiGoals

H, TOL, PRE_SEGS, ADR_EPS = 25, 3.0, 2, 0.01
MEAN, STD = get_normalizer_preset('ant_preset')
CTRLS = ['bearing1', 'bearing25', 'adr', 'bearing1_stop', 'bearing25_stop', 'adr_nostop']


def unit(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


def adr_horizon(dist, vhat, tol):
    """h_k of ADR. vhat = nan -> h_max; h = 0 inside the success ball (freeze)."""
    h = np.floor(np.maximum(dist - tol, 0) / (2 * np.nan_to_num(vhat) + ADR_EPS))
    return np.where(np.isnan(vhat), H, np.clip(h, 0, H)).astype(int)


class Skills:
    def __init__(self, run, epoch):
        self.policy = torch.load(f'{run}/option_policy{epoch}.pt', map_location='cpu')['policy']
        self.policy._force_use_mode_actions = True
        self.phi_net = torch.load(f'{run}/traj_encoder{epoch}.pt', map_location='cpu')['traj_encoder']
        self.ray = hasattr(self.policy, 'ray_phi')

    def phi(self, obs):
        with torch.no_grad():
            return self.phi_net(torch.as_tensor(obs, dtype=torch.float32)).mean.numpy()

    def reset_rows(self, obs, mask):
        """Start a new skill (u_t = 0) in the rows `mask` only."""
        if not self.ray or not mask.any():
            return
        with torch.no_grad():
            phi = self.policy.ray_phi(torch.as_tensor(obs, dtype=torch.float32)).mean
        if self.policy._phi_start is None or mask.all():
            self.policy._phi_start = phi
        else:
            m = torch.as_tensor(mask)
            self.policy._phi_start[m] = phi[m]

    def act(self, obs, z):
        return self.policy.get_actions(np.concatenate([obs, z], 1).astype(np.float32))[0]


def goal_obs(envs, goals):
    raw = np.stack([e._cur_obs.copy() for e in envs])
    raw[:, :2] = goals
    return (raw - MEAN) / (STD + 1e-8)


def xy_of(envs):
    return np.stack([e.unwrapped.sim.data.qpos.flat[:2].copy() for e in envs])


def moving_start(sk, n, budget, controller):
    envs = [consistent_normalize(AntEnv(render_hw=100), normalize_obs=True, mean=MEAN, std=STD) for _ in range(n)]
    obs = []
    for e in envs:
        np.random.seed(0)
        obs.append(e.reset())
    obs = np.stack(obs)
    # pre-roll: goal-bearing skill towards a point 20 m ahead (+x), PRE_SEGS segments of H steps
    pre_goal = xy_of(envs) + np.array([20., 0.])
    phi_pre = sk.phi(goal_obs(envs, pre_goal))
    for _ in range(PRE_SEGS):
        phi_seg0 = sk.phi(obs)
        xy_seg0 = xy_of(envs)
        z = unit(phi_pre - phi_seg0)
        sk.reset_rows(obs, np.ones(n, bool))
        for _ in range(H):
            obs = np.stack([e.step(a)[0] for e, a in zip(envs, sk.act(obs, z))])
    p = xy_of(envs)
    t_ang = np.arctan2(*(p - xy_seg0)[:, ::-1].T)
    a = np.radians(-180 + 360 * (np.arange(n) + 1) / n)  # relative goal angle, evenly spaced, includes 180
    r = 10 + 30 * ((np.arange(n) * 0.6180339887498949) % 1)
    goals = p + r[:, None] * np.stack([np.cos(t_ang + a), np.sin(t_ang + a)], 1)
    phi_g = sk.phi(goal_obs(envs, goals))

    xy = [p]
    countdown = np.zeros(n, int)
    z = np.zeros((n, 2))
    # ADR: seed speed from the last pre-roll segment (already observed motion), not a cold h_max.
    seg_xy0, seg_h, vhat, horizons = np.zeros((n, 2)), np.zeros(n, int), np.linalg.norm(p - xy_seg0, axis=1) / H, []
    freeze = np.zeros(n, bool)
    selections = np.zeros(n, int)
    stopped = np.zeros(n, bool)
    for step in range(budget):
        cur = xy[-1]
        d = unit(phi_g - sk.phi(obs))
        dist = np.linalg.norm(goals - cur, axis=1)
        if controller in ('bearing1', 'bearing1_stop'):
            if step == 0:
                sk.reset_rows(obs, np.ones(n, bool))
            if controller == 'bearing1_stop':
                stopped = dist <= TOL
            z = d
            selections += ~stopped
        elif controller == 'bearing25':
            if step % H == 0:
                selections += 1
                z = d
                if step == 0:
                    sk.reset_rows(obs, np.ones(n, bool))
        elif controller == 'bearing25_stop':
            was = stopped
            stopped = dist <= TOL
            due = ((countdown <= 0) | (was & ~stopped)) & ~stopped
            if step == 0:
                sk.reset_rows(obs, np.ones(n, bool))
            z = np.where(due[:, None], d, z)
            countdown[due] = H
            selections += due
            countdown -= 1
        elif controller in ('adr', 'adr_nostop'):
            due = countdown <= 0
            if due.any():
                moved = due & (seg_h > 0) & ~freeze  # only update speed after a real walking segment
                if moved.any():
                    vhat[moved] = np.linalg.norm(cur[moved] - seg_xy0[moved], axis=1) / seg_h[moved]
                h = adr_horizon(dist[due], vhat[due], TOL)
                if controller == 'adr_nostop':
                    h = np.maximum(h, 1)  # at least one step, never frozen
                sel = np.zeros(n, bool)
                sel[due] = h > 0
                selections += sel
                horizons.append(h)
                z = np.where(due[:, None], d, z)
                if step == 0:
                    sk.reset_rows(obs, due)
                freeze[due] = h == 0
                seg_xy0[due], seg_h[due] = cur[due], np.maximum(h, 0)
                countdown[due] = np.where(h == 0, 1, h)  # recheck every step while frozen
            countdown -= 1
        else:
            raise ValueError(controller)
        acts = sk.act(obs, z)
        if controller == 'adr' and freeze.any():
            acts = np.where(freeze[:, None], 0.0, acts)  # target length 0 inside the success ball
        if controller.endswith('_stop') and stopped.any():
            acts = np.where(stopped[:, None], 0.0, acts)  # stop: zero action inside the success ball
        obs = np.stack([e.step(x)[0] for e, x in zip(envs, acts)])
        xy.append(xy_of(envs))
    out = dict(xy=np.stack(xy, 1).astype(np.float32), goals=goals, rel_angle=np.degrees(a), radius=r,
               selections=selections)
    if controller in ('adr', 'adr_nostop'):
        out['horizons'] = np.concatenate(horizons)
    return out


def multigoals_adr(env):
    """One AntMultiGoals episode, primitive-step loop of MultiGoals.step with z = d replanned on the ADR schedule."""
    n = env.n
    ret, z = np.zeros(n), np.zeros((n, 2))
    # Keep the speed estimate across goals (motion speed does not reset); only the first goal starts cold.
    countdown, seg_h, seg_xy0 = np.zeros(n, int), np.zeros(n, int), np.zeros((n, 2))
    vhat, freeze = np.full(n, np.nan), np.zeros(n, bool)
    goal_prev = env.goal.copy()
    while env.t < EP_LEN:
        if env.t % K == 0:
            env.policy.reset()
        cur = env.xy()
        new_goal = np.any(env.goal != goal_prev, 1)
        goal_prev = env.goal.copy()
        due = (countdown <= 0) | new_goal
        if due.any():
            moved = due & (seg_h > 0) & ~freeze & ~new_goal
            if moved.any():
                vhat[moved] = np.linalg.norm(cur[moved] - seg_xy0[moved], axis=1) / seg_h[moved]
            h = adr_horizon(np.linalg.norm(env.goal[due] - cur[due], axis=1), vhat[due], RADIUS)
            z = np.where(due[:, None], env.reference_z(), z)
            freeze[due] = h == 0
            seg_xy0[due], seg_h[due] = cur[due], np.maximum(h, 0)
            countdown[due] = np.where(h == 0, 1, h)
        actions, _ = env.policy.get_actions(np.concatenate([env.obs, z], 1))
        if freeze.any():
            actions = np.where(freeze[:, None], 0.0, actions)
        env.obs = np.stack([e.step(a)[0] for e, a in zip(env.envs, actions)])
        countdown -= 1
        env.t += 1
        active = env.goals_done < N_GOALS
        env.goal_t += active
        reached = active & (np.linalg.norm(env.xy() - env.goal, axis=1) <= RADIUS)
        timeout = active & ~reached & (env.goal_t >= GOAL_STEPS)
        ret += GOAL_REWARD * reached
        new = reached | timeout
        env.goals_done += new
        env.goal[new] = env.xy()[new] + env.rng.uniform(-GOAL_RANGE, GOAL_RANGE, (new.sum(), 2))
        env.goal_t[new] = 0
        freeze[new] = False  # a new goal unfreezes; speed estimate is kept
    return ret


def multigoals(sk, n_envs, episodes):
    np.random.seed(0)  # the same env reset noise for every run
    env = MultiGoals(n_envs, sk.policy, sk.phi_net, seed=10_000)
    returns = []
    while len(returns) < episodes:
        env.reset()
        returns.extend(multigoals_adr(env).tolist())
    return np.array(returns[:episodes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='training run directory (with option_policy<epoch>.pt)')
    ap.add_argument('--label', required=True, help='output name, e.g. dart_s0 (see eval/ant/summarize.py)')
    ap.add_argument('--epoch', type=int, default=10000)
    ap.add_argument('--n', type=int, default=256, help='moving-start goals')
    ap.add_argument('--budget', type=int, default=400, help='goal-phase steps')
    ap.add_argument('--ctrls', nargs='+', default=CTRLS, choices=CTRLS)
    ap.add_argument('--mg_envs', type=int, default=32)
    ap.add_argument('--mg_episodes', type=int, default=256)
    ap.add_argument('--no_mg', action='store_true', help='skip AntMultiGoals')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'ant', 'goal_reach'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sk = Skills(args.run, args.epoch)
    res = {}
    for c in args.ctrls:
        r = moving_start(sk, args.n, args.budget, c)
        dist = np.linalg.norm(r['xy'] - r['goals'][:, None], axis=2)
        print(f'{args.label} {c:14s} final {np.mean(dist[:, -1] <= TOL):.3f} reach {np.mean(dist.min(1) <= TOL):.3f} '
              f'dist {dist[:, -1].mean():5.2f} selections {r["selections"].mean():.1f}', flush=True)
        for k, v in r.items():
            res[f'{c}_{k}'] = v
    if not args.no_mg:
        ret = multigoals(sk, args.mg_envs, args.mg_episodes)
        print(f'{args.label} AntMultiGoals ADR return {ret.mean():.3f} +- {ret.std() / np.sqrt(len(ret)):.3f}', flush=True)
        res['mg_adr'] = ret
    np.savez_compressed(f'{args.out}/{args.label}.npz', **res)
    print('saved', f'{args.out}/{args.label}.npz')


if __name__ == '__main__':
    main()
