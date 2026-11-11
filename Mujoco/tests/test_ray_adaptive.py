"""Fast CPU tests for the adaptive ray mode (iod/metra.py, --reward_mode ray --ray_beta_mode adaptive).

    python tests/test_ray_adaptive.py
"""
import os
import sys
import types

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'garaged', 'src')]

import dowel_wrapper  # noqa: F401  (must come before dowel / garage)
import numpy as np
import torch
import torch.nn as nn

from iod.metra import METRA

OBS, WARMUP, BETA = 5, (10, 20), 0.5


class FirstTwo(nn.Module):
    """phi(s) = the first two state dims, so u_T = s_T[:2] - s_0[:2]."""

    def forward(self, x):
        return types.SimpleNamespace(mean=x[:, :2])


def fake_algo(epoch=0, window=1):
    algo = object.__new__(METRA)
    algo.device, algo.dim_option, algo.ray_beta_mode = torch.device('cpu'), 2, 'adaptive'
    algo.ray_beta, algo.ray_warmup, algo._epoch = BETA, WARMUP, epoch
    algo.traj_encoder = FirstTwo()
    algo._init_ray_adaptive(window)
    return algo


def unit(deg):
    deg = np.radians(np.asarray(deg, float))
    return np.stack([np.cos(deg), np.sin(deg)], -1)


def tz(deg):
    return torch.as_tensor(unit(deg), dtype=torch.float32).reshape(-1, 2)


def paths(z_deg, end_deg, T=4, length=100.):
    """One trajectory per skill direction z_deg, ending `length` phi units away at direction end_deg."""
    data = {'obs': [], 'next_obs': [], 'options': []}
    for zd, ed in zip(z_deg, end_deg):
        obs = np.zeros((T, OBS))
        next_obs = np.zeros((T, OBS))
        next_obs[-1, :2] = length * unit(ed)
        data['obs'].append(obs)
        data['next_obs'].append(next_obs)
        data['options'].append(np.repeat(unit(zd)[None], T, 0))
    return data


def test_equals_fixed_beta_before_updates():
    z = tz(np.random.RandomState(0).uniform(0, 360, 50))
    for epoch, ramp in [(0, 0.), (5, 0.), (15, 0.5), (20, 1.), (500, 1.)]:
        algo = fake_algo(epoch)
        assert torch.allclose(algo._ray_beta_now(z), torch.full((50,), BETA * ramp), atol=1e-6), epoch
        assert algo._ray_beta_now() == BETA * ramp  # without z: the fixed value


def test_paper_rule():
    """Errors 0, sin^2(30 deg) = 0.25, 0, 0 at 0, 90, 180, 270 deg: e(psi) is a triangle of height 0.25 between 0 and
    180 deg, mean 0.25 * 90 / 360 = 0.0625, so beta = 0.5 (1 + e) / 1.0625."""
    algo = fake_algo(epoch=500)
    stats = algo._update_ray_adaptive(paths([0, 90, 180, 270], [0, 150, 180, 270]))
    expect = {0: 0., 45: 0.125, 90: 0.25, 135: 0.125, 180: 0., 225: 0., 315: 0.}
    for deg, e in expect.items():
        got = algo._ray_beta_now(tz(deg)).item()
        assert abs(got - BETA * (1 + e) / 1.0625) < 1e-4, (deg, got)
    assert abs(algo._ray_beta_grid.mean().item() - BETA) < 1e-5  # mean beta preserved
    assert abs(stats['RayEndAngleMax'].item() - 60) < 1e-3 and stats['RayAdaptTrajs'].item() == 4


def test_bounds_and_mean():
    """Random errors, including motion opposite to the command (e = 1): beta in [beta / 2, 2 beta], mean beta."""
    rs = np.random.RandomState(1)
    z = rs.uniform(0, 360, 8)
    end = z + rs.choice([0, 180, 20, -90], 8)
    algo = fake_algo(epoch=500)
    algo._update_ray_adaptive(paths(z, end))
    g = algo._ray_beta_grid
    assert g.min().item() >= BETA / 2 - 1e-6 and g.max().item() <= 2 * BETA + 1e-6
    assert abs(g.mean().item() - BETA) < 1e-5


def test_periodic_interpolation():
    """Points at 350 and 10 deg: beta at 0 deg interpolates across the wrap."""
    algo = fake_algo(epoch=500)
    algo._update_ray_adaptive(paths([350, 10, 180], [350 + 60, 10, 180]))  # errors 0.25, 0, 0
    b350, b0, b10 = (algo._ray_beta_now(tz(d)).item() for d in [350, 0, 10])
    assert abs(b0 - (b350 + b10) / 2) < 1e-4 and b350 > b0 > b10, (b350, b0, b10)


def test_window_and_latest_phi():
    """Window 1 uses only the latest epoch; window 2 pools two epochs."""
    first, second = paths([90], [150]), paths([0, 180], [0, 180])
    algo1, algo2 = fake_algo(500, window=1), fake_algo(500, window=2)
    for algo in (algo1, algo2):
        algo._update_ray_adaptive(first)
        algo._update_ray_adaptive(second)
    assert torch.allclose(algo1._ray_beta_grid, torch.full((len(algo1._ray_beta_grid),), BETA))  # errors all 0
    assert algo2._ray_beta_now(tz(90)).item() > BETA > algo2._ray_beta_now(tz(0)).item()


def test_no_valid_displacement_keeps_beta():
    algo = fake_algo(epoch=500)
    algo._update_ray_adaptive(paths([0, 90, 180, 270], [0, 150, 180, 270]))
    before = algo._ray_beta_grid.clone()
    algo._update_ray_adaptive(paths([45, 200], [45, 200], length=0.))
    assert torch.equal(algo._ray_beta_grid, before)


def test_reward_uses_beta_of_each_skill():
    """The SAC penalty term scales with beta(z) of each sample's own skill (warmup ramp included)."""
    algo = fake_algo(epoch=15)  # ramp 0.5
    algo._update_ray_adaptive(paths([0, 90, 180, 270], [0, 150, 180, 270]))
    z = tz([0, 90])
    got = algo._ray_beta_now(z)
    expect = 0.5 * BETA * torch.tensor([1., 1.25]) / 1.0625
    assert torch.allclose(got, expect, atol=1e-4), got


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
