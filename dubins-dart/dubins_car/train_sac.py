# -*- coding: utf-8 -*-
"""Skill-conditioned SAC for the Dubins car.

The finite-horizon objective is the undiscounted sum of per-step rewards, so
gamma is 1 and the episode terminates at the skill horizon. The observation
includes remaining-time fraction because the optimal turn depends on how many
steps are left. phi is fixed to position, so only the policy is trained.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dubins_car.adaptive import AdaptiveBeta, AdaptiveConeAlpha
from dubins_car.d3_schedule import D3MixSchedule
from dubins_car.dynamics import DT, OBS_DIM, OMEGA_MAX, V, horizon_steps
from dubins_car.metrics import alignment_bound_rad, angular_gap, direction_errors
from dubins_car.torch_dyn import make_obs_torch, reward_torch, step_dynamics_torch

# Modes that freeze a direction-dependent absolute lateral weight per skill.
ADAPTIVE_MODES = ("adaptive", "cone")


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, -0.5)

    def forward(self, obs):
        h = self.body(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        return mean, log_std.exp()

    def sample(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        # Stable tanh correction: log(1 - tanh(x)^2) = 2 (log 2 - x - softplus(-2x)).
        log_prob = dist.log_prob(raw) - (2 * (math.log(2) - raw - F.softplus(-2 * raw)))
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.q1 = _mlp(obs_dim + 1, hidden, 1)
        self.q2 = _mlp(obs_dim + 1, hidden, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class Replay:
    def __init__(self, capacity, obs_dim, device, store_geometry: bool = False):
        self.capacity = int(capacity)
        self.device = device
        self.store_geometry = store_geometry
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, 1, device=device)
        self.rew = torch.zeros(capacity, 1, device=device)
        self.done = torch.zeros(capacity, 1, device=device)
        if store_geometry:
            self.z = torch.zeros(capacity, 2, device=device)
            self.u_old = torch.zeros(capacity, 2, device=device)
            self.u_new = torch.zeros(capacity, 2, device=device)
        self.ptr = 0
        self.count = 0

    def add(self, obs, act, rew, nxt, done, z=None, u_old=None, u_new=None):
        act = act.reshape(-1, 1)
        rew = rew.reshape(-1, 1)
        done = done.reshape(-1, 1).float()
        n = obs.shape[0]
        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self._write(sl, obs, act, rew, nxt, done, z, u_old, u_new)
            self.ptr = end % self.capacity
        else:
            first = self.capacity - self.ptr
            self._write(
                slice(self.ptr, self.capacity),
                obs[:first],
                act[:first],
                rew[:first],
                nxt[:first],
                done[:first],
                None if z is None else z[:first],
                None if u_old is None else u_old[:first],
                None if u_new is None else u_new[:first],
            )
            rest = n - first
            self._write(
                slice(0, rest),
                obs[first:],
                act[first:],
                rew[first:],
                nxt[first:],
                done[first:],
                None if z is None else z[first:],
                None if u_old is None else u_old[first:],
                None if u_new is None else u_new[first:],
            )
            self.ptr = rest
        self.count = min(self.capacity, self.count + n)

    def _write(self, sl, obs, act, rew, nxt, done, z, u_old, u_new):
        self.obs[sl] = obs
        self.act[sl] = act
        self.rew[sl] = rew
        self.next[sl] = nxt
        self.done[sl] = done
        if self.store_geometry and z is not None:
            self.z[sl] = z
            self.u_old[sl] = u_old
            self.u_new[sl] = u_new

    def sample(self, batch):
        idx = torch.randint(0, self.count, (batch,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next[idx], self.done[idx]

    def recompute_dart_rewards(self, schedule):
        """Replay rewards use the current direction-dependent penalty."""
        if not self.store_geometry or self.count == 0:
            return
        n = self.count
        beta = schedule.beta_for_z(self.z[:n])
        rew = reward_torch(self.u_old[:n], self.u_new[:n], self.z[:n], "dart", beta)
        self.rew[:n] = rew.unsqueeze(-1)


class SAC:
    def __init__(self, obs_dim, hidden, lr, gamma, tau, device):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -1.0
        self.actor = Actor(obs_dim, hidden).to(device)
        self.critic = Critic(obs_dim, hidden).to(device)
        self.critic_t = Critic(obs_dim, hidden).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_t.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.1), device=device))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

    def act(self, obs):
        action, _, _ = self.actor.sample(obs)
        return action.squeeze(-1)

    def update(self, replay, batch):
        obs, act, rew, nxt, done = replay.sample(batch)
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(nxt)
            q1_t, q2_t = self.critic_t(nxt, next_a)
            q_t = torch.min(q1_t, q2_t) - alpha.detach() * next_logp
            backup = rew + self.gamma * (1.0 - done) * q_t
        q1, q2 = self.critic(obs, act)
        loss_c = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_a, logp, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_a)
        loss_a = (alpha.detach() * logp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
                tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        return loss_c.detach(), alpha.detach()


class BatchCar:
    def __init__(
        self,
        n,
        horizon_radii,
        mode,
        weight,
        device,
        schedule=None,
        d3_schedule: D3MixSchedule | None = None,
    ):
        self.n = n
        self.mode = mode
        self.weight = float(weight)
        self.device = device
        self.schedule = schedule
        self.d3_schedule = d3_schedule
        self.horizon = horizon_steps(horizon_radii)
        self.path_length = self.horizon * DT * V
        self.state = torch.zeros(n, 3, device=device)
        self.u = torch.zeros(n, 2, device=device)
        self.t = torch.zeros(n, dtype=torch.long, device=device)
        self.z = self._sample_z(n)
        self.ep_ret = torch.zeros(n, device=device)
        self.beta = self._beta_for_current_z()

    def _sample_z(self, n):
        ang = torch.rand(n, device=self.device) * (2 * math.pi)
        return torch.stack((ang.cos(), ang.sin()), dim=-1)

    def _beta_for_current_z(self):
        if self.mode in ADAPTIVE_MODES and self.schedule is not None:
            return self.schedule.beta_for_z(self.z)
        return torch.full((self.n,), self.weight, device=self.device)

    def _reward_mode(self):
        # Cone schedule still uses absolute-lateral DART reward; only beta(z) differs.
        if self.mode in ADAPTIVE_MODES:
            return "dart"
        return self.mode

    def obs(self):
        return make_obs_torch(self.state, self.z, self.u, self.t, self.horizon, self.path_length)

    def step(self, action):
        u_old = self.u.clone()
        self.state = step_dynamics_torch(self.state, action)
        self.u = self.state[:, :2]
        weight = self.beta if self.mode in ADAPTIVE_MODES else self.weight
        if self.mode == "d3":
            alpha = 0.0 if self.d3_schedule is None else self.d3_schedule.alpha
            reward = reward_torch(u_old, self.u, self.z, self.mode, weight, alpha=alpha)
            if self.d3_schedule is not None:
                self.d3_schedule.observe(u_old, self.u, self.z)
        else:
            reward = reward_torch(u_old, self.u, self.z, self._reward_mode(), weight)
        self.t = self.t + 1
        done = self.t >= self.horizon
        return self.obs(), reward, done, u_old, self.u.clone(), self.z.clone()

    def reset_done(self, done):
        new_z = self._sample_z(self.n)
        zeros_s = torch.zeros_like(self.state)
        zeros_u = torch.zeros_like(self.u)
        self.state = torch.where(done.unsqueeze(-1), zeros_s, self.state)
        self.u = torch.where(done.unsqueeze(-1), zeros_u, self.u)
        self.z = torch.where(done.unsqueeze(-1), new_z, self.z)
        self.t = torch.where(done, torch.zeros_like(self.t), self.t)
        # Freeze the skill's beta for the whole execution; refresh only on reset.
        new_beta = self._beta_for_current_z()
        self.beta = torch.where(done, new_beta, self.beta)


@torch.no_grad()
def rollout_deterministic(actor, z, horizon, path_length, device):
    b = z.shape[0]
    state = torch.zeros(b, 3, device=device)
    u = torch.zeros(b, 2, device=device)
    t = torch.zeros(b, dtype=torch.long, device=device)
    for _ in range(horizon):
        obs = make_obs_torch(state, z, u, t, horizon, path_length)
        action = actor.deterministic(obs).squeeze(-1)
        state = step_dynamics_torch(state, action)
        u = state[:, :2]
        t = t + 1
    return u


def evaluate(actor, horizon, path_length, device, k=360):
    psi = torch.arange(k, device=device, dtype=torch.float32) * (2 * math.pi / k)
    z = torch.stack((psi.cos(), psi.sin()), dim=-1)
    u = rollout_deterministic(actor, z, horizon, path_length, device)
    z_np = z.cpu().numpy()
    u_np = u.cpu().numpy()
    err = direction_errors(z_np, u_np)
    gap = angular_gap(u_np)
    return {
        "max_err_deg": float(np.degrees(err.max())),
        "mean_err_deg": float(np.degrees(err.mean())),
        "blind_deg": gap["gap_deg"],
        "blind_center_deg": gap["center_deg"],
        "mean_forward": float(np.sum(z_np * u_np, axis=1).mean()),
        "u": u_np,
        "z": z_np,
        "psi": psi.cpu().numpy(),
        "err": err,
    }


def train_one(
    horizon_radii,
    mode,
    weight,
    device,
    total_steps=100_000,
    num_envs=16,
    batch_size=256,
    hidden=256,
    lr=3e-4,
    gamma=1.0,
    tau=0.005,
    learning_starts=2_000,
    eval_every=20_000,
    seed=0,
    L=float("nan"),
    log=print,
    adaptive_update_every=2_000,
    updates_per_step=None,
    eta=1.0,
    theta_tol=None,
    kappa=0.5,
    alpha_max_deg=70.0,
    only_increase=False,
    n_bins=72,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    total_steps = int(math.ceil(total_steps / float(num_envs)) * num_envs)
    if mode == "adaptive":
        schedule = AdaptiveBeta(weight, device=device, eta=eta)
    elif mode == "cone":
        if theta_tol is None:
            raise ValueError("mode='cone' requires theta_tol (radians)")
        schedule = AdaptiveConeAlpha(
            beta_base=weight,
            theta_tol=float(theta_tol),
            n_bins=n_bins,
            device=device,
            kappa=kappa,
            alpha_max=math.radians(float(alpha_max_deg)),
            only_increase=only_increase,
        )
    else:
        schedule = None
    d3_schedule = D3MixSchedule(device=device) if mode == "d3" else None
    env = BatchCar(
        num_envs, horizon_radii, mode, weight, device, schedule=schedule, d3_schedule=d3_schedule
    )
    agent = SAC(OBS_DIM, hidden, lr, gamma, tau, device)
    replay = Replay(total_steps, OBS_DIM, device, store_geometry=(mode in ADAPTIVE_MODES))
    obs = env.obs()
    steps = 0
    finished_sum = torch.zeros((), device=device)
    finished_count = torch.zeros((), device=device)
    history = {
        "steps": [],
        "max_err_deg": [],
        "mean_err_deg": [],
        "blind_deg": [],
        "mean_forward": [],
        "return": [],
        "mean_beta": [],
        "mean_e": [],
        "mean_alpha_deg": [],
        "mean_theta_deg": [],
        "d3_alpha": [],
        "d3_cos": [],
    }
    next_eval = eval_every
    t0 = time.time()
    last_loss = None
    last_alpha = None
    pending_z = []
    pending_u = []
    next_adaptive = adaptive_update_every
    n_grad = env.n if updates_per_step is None else int(updates_per_step)
    beta_for_bound = 0.0 if mode == "metra" else (weight if mode in ("dart", "adaptive", "cone") else float("nan"))
    bound_deg = (
        math.degrees(alignment_bound_rad(L, beta_for_bound))
        if mode in ("metra", "dart", "adaptive", "cone") and L == L
        else float("nan")
    )
    if mode == "cone":
        log(
            f"  cone schedule: alpha0={math.degrees(schedule.alpha0):.2f}deg  "
            f"theta_tol={math.degrees(schedule.theta_tol):.2f}deg  "
            f"kappa={schedule.kappa}  only_increase={schedule.only_increase}"
        )

    while steps < total_steps:
        if steps < learning_starts:
            action = torch.empty(env.n, device=device).uniform_(-1, 1)
        else:
            with torch.no_grad():
                action = agent.act(obs)
        next_obs, reward, done, u_old, u_new, z = env.step(action)
        if mode in ADAPTIVE_MODES:
            replay.add(obs, action, reward, next_obs, done, z=z, u_old=u_old, u_new=u_new)
        else:
            replay.add(obs, action, reward, next_obs, done)
        env.ep_ret = env.ep_ret + reward
        finished_sum = finished_sum + torch.where(done, env.ep_ret, torch.zeros_like(env.ep_ret)).sum()
        finished_count = finished_count + done.float().sum()
        if mode in ADAPTIVE_MODES and bool(done.any()):
            pending_z.append(z[done].detach())
            pending_u.append(u_new[done].detach())
        env.ep_ret = torch.where(done, torch.zeros_like(env.ep_ret), env.ep_ret)
        env.reset_done(done)
        obs = env.obs()
        steps += env.n
        if mode in ADAPTIVE_MODES and pending_z and steps >= next_adaptive:
            z_cat = torch.cat(pending_z, dim=0)
            u_cat = torch.cat(pending_u, dim=0)
            stats = schedule.update_from_endpoints(z_cat, u_cat)
            replay.recompute_dart_rewards(schedule)
            pending_z.clear()
            pending_u.clear()
            while next_adaptive <= steps:
                next_adaptive += adaptive_update_every
            if stats["updated"]:
                if mode == "cone":
                    log(
                        f"  cone update@{steps}: n={stats['n_obs']} bins={stats['n_bins_updated']}  "
                        f"theta={math.degrees(stats['mean_theta']):.2f}deg  "
                        f"alpha=[{stats.get('min_alpha_deg', float('nan')):.1f}, "
                        f"{stats.get('max_alpha_deg', float('nan')):.1f}] "
                        f"mean={stats['mean_alpha_deg']:.1f}deg  "
                        f"beta_mean={stats['mean_beta']:.3f}"
                    )
                else:
                    log(
                        f"  adaptive update@{steps}: n={stats['n_obs']}  "
                        f"mean_e={stats['mean_e']:.3f}  beta=[{stats['min_beta']:.3f}, {stats['max_beta']:.3f}] "
                        f"mean={stats['mean_beta']:.3f}"
                    )
        if steps >= learning_starts:
            for _ in range(n_grad):
                last_loss, last_alpha = agent.update(replay, batch_size)
        crossed = steps // 2000 != (steps - env.n) // 2000
        if crossed and last_loss is not None:
            if not math.isfinite(float(last_loss)):
                raise RuntimeError("SAC critic loss became NaN")
            ret = (finished_sum / finished_count.clamp(min=1)).item()
            elapsed = time.time() - t0
            log(
                f"  steps={steps}  return={ret:.3f}  "
                f"critic={float(last_loss):.4f}  alpha={float(last_alpha):.3f}  "
                f"{steps / elapsed:.0f} steps/s"
            )
            finished_sum = torch.zeros((), device=device)
            finished_count = torch.zeros((), device=device)
        if steps >= next_eval or steps >= total_steps:
            ev = evaluate(agent.actor, env.horizon, env.path_length, device)
            history["steps"].append(steps)
            history["max_err_deg"].append(ev["max_err_deg"])
            history["mean_err_deg"].append(ev["mean_err_deg"])
            history["blind_deg"].append(ev["blind_deg"])
            history["mean_forward"].append(ev["mean_forward"])
            history["return"].append(
                float((finished_sum / finished_count.clamp(min=1)).item()) if finished_count.item() else float("nan")
            )
            if schedule is not None:
                # Refresh schedule using deterministic endpoints so it tracks the policy.
                z_eval = torch.as_tensor(ev["z"], device=device, dtype=torch.float32)
                u_eval = torch.as_tensor(ev["u"], device=device, dtype=torch.float32)
                stats = schedule.update_from_endpoints(z_eval, u_eval)
                replay.recompute_dart_rewards(schedule)
                history["mean_beta"].append(stats["mean_beta"])
                history["mean_e"].append(stats["mean_e"])
                history["mean_alpha_deg"].append(float(stats.get("mean_alpha_deg", float("nan"))))
                history["mean_theta_deg"].append(
                    float(math.degrees(stats["mean_theta"])) if "mean_theta" in stats else float("nan")
                )
                history["d3_alpha"].append(float("nan"))
                history["d3_cos"].append(float("nan"))
                if mode == "cone":
                    log(
                        f"  eval steps={steps}  max_err={ev['max_err_deg']:.2f} deg  "
                        f"mean_err={ev['mean_err_deg']:.2f}  blind={ev['blind_deg']:.2f}  "
                        f"forward={ev['mean_forward']:.2f}  "
                        f"alpha_mean={stats['mean_alpha_deg']:.1f}deg  "
                        f"theta={math.degrees(stats['mean_theta']):.2f}deg  "
                        f"beta_mean={stats['mean_beta']:.3f}"
                    )
                else:
                    log(
                        f"  eval steps={steps}  max_err={ev['max_err_deg']:.2f} deg  "
                        f"mean_err={ev['mean_err_deg']:.2f}  blind={ev['blind_deg']:.2f}  "
                        f"forward={ev['mean_forward']:.2f}  bound={bound_deg:.2f}  "
                        f"beta_mean={stats['mean_beta']:.3f}  e={stats['mean_e']:.3f}"
                    )
            elif d3_schedule is not None:
                snap = d3_schedule.snapshot()
                history["mean_beta"].append(snap["alpha"])
                history["mean_e"].append(snap["ema_cos"])
                history["mean_alpha_deg"].append(float("nan"))
                history["mean_theta_deg"].append(float("nan"))
                history["d3_alpha"].append(snap["alpha"])
                history["d3_cos"].append(snap["ema_cos"])
                log(
                    f"  eval steps={steps}  max_err={ev['max_err_deg']:.2f} deg  "
                    f"mean_err={ev['mean_err_deg']:.2f}  blind={ev['blind_deg']:.2f}  "
                    f"forward={ev['mean_forward']:.2f}  "
                    f"d3_alpha={snap['alpha']:.3f}  cos={snap['ema_cos']:.3f}"
                )
            else:
                history["mean_beta"].append(weight if mode == "dart" else 0.0)
                history["mean_e"].append(float("nan"))
                history["mean_alpha_deg"].append(float("nan"))
                history["mean_theta_deg"].append(float("nan"))
                history["d3_alpha"].append(float("nan"))
                history["d3_cos"].append(float("nan"))
                log(
                    f"  eval steps={steps}  max_err={ev['max_err_deg']:.2f} deg  "
                    f"mean_err={ev['mean_err_deg']:.2f}  blind={ev['blind_deg']:.2f}  "
                    f"forward={ev['mean_forward']:.2f}  bound={bound_deg:.2f}"
                )
            while next_eval <= steps:
                next_eval += eval_every
            final_eval = ev
    final_eval["history"] = history
    final_eval["seconds"] = time.time() - t0
    final_eval["bound_deg"] = bound_deg
    final_eval["actor"] = agent.actor
    if schedule is not None:
        final_eval["adaptive"] = schedule.snapshot()
    if d3_schedule is not None:
        final_eval["d3"] = d3_schedule.snapshot()
    return final_eval
