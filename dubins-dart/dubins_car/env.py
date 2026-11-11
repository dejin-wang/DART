# -*- coding: utf-8 -*-
"""Gymnasium Dubins car with a fixed position representation."""

from __future__ import annotations

import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from dubins_car.dynamics import (
    DT,
    OBS_DIM,
    OMEGA_MAX,
    V,
    horizon_steps,
    integrate_step_np,
    make_obs_np,
    reward_np,
)

_REGISTERED = False


def register_dubins_car() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        register(id="DubinsCar-v0", entry_point="dubins_car.env:DubinsCarEnv")
    except Exception:
        pass
    _REGISTERED = True


class DubinsCarEnv(gym.Env):
    """Constant-speed car. Action in [-1, 1] scales the turning rate.

    The episode is one skill execution. The code z is fixed until reset.
    Reward modes:
      metra   F = z · u
      dart    F = z · u - beta * lateral(u)
      squared F = z · u - lambda * lateral(u)^2
      norm    F = (1+lambda) z · u - lambda * ||u||
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon_radii: float = 10.0,
        reward_mode: str = "metra",
        weight: float = 0.0,
        dt: float = DT,
        v: float = V,
        omega_max: float = OMEGA_MAX,
    ):
        super().__init__()
        self.horizon_radii = float(horizon_radii)
        self.reward_mode = reward_mode
        self.weight = float(weight)
        self.dt = float(dt)
        self.v = float(v)
        self.omega_max = float(omega_max)
        self.horizon = horizon_steps(self.horizon_radii, self.dt, self.v, self.omega_max)
        self.path_length = self.horizon * self.dt * self.v
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float64)
        self.u = np.zeros(2, dtype=np.float64)
        self.z = np.array([1.0, 0.0])
        self.t = 0

    def _sample_z(self) -> np.ndarray:
        psi = float(self.np_random.random() * 2 * math.pi)
        return np.array([math.cos(psi), math.sin(psi)], dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state[:] = 0.0
        self.u[:] = 0.0
        self.t = 0
        if options is not None and "z" in options:
            z = np.asarray(options["z"], dtype=np.float64).reshape(2)
            n = float(np.linalg.norm(z))
            if n < 1e-8:
                raise ValueError("skill z must be nonzero")
            self.z = z / n
        else:
            self.z = self._sample_z()
        return self._obs(), {"z": self.z.copy()}

    def _obs(self) -> np.ndarray:
        return make_obs_np(self.state, self.z, self.u, self.t, self.horizon, self.path_length)

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        u_old = self.u.copy()
        self.state = integrate_step_np(self.state, a, self.dt, self.v, self.omega_max)
        self.u = self.state[:2].copy()
        reward = reward_np(u_old, self.u, self.z, self.reward_mode, self.weight)
        self.t += 1
        terminated = self.t >= self.horizon
        info = {"u": self.u.copy(), "phi": self.u.copy(), "z": self.z.copy()}
        return self._obs(), float(reward), bool(terminated), False, info


register_dubins_car()
