#!/usr/bin/env python3
"""HalfCheetah result tables and diagnostic figure from results/half_cheetah/ (mean ± s.d. over training seeds).

Labels are <method>_s<seed>, method in METHODS below. Inputs (any missing part is skipped):
  results/half_cheetah/protocol/<label>.npz        hc_protocol_eval.py            exploration, behaviour, HalfCheetahGoal (h = 1)
  results/half_cheetah/goal_ctrl/<label>_free.npz  hc_goal_controllers.py --mode free   h = 1 / h = 25 / ADR
  results/half_cheetah/goal_ctrl/<label>_sac.npz   hc_goal_controllers.py --mode sac    high-level SAC

Exploration (360 evenly spaced z, 200 steps, fixed start): mean |x_T|; visited 1 m x-bins (METRA's HalfCheetah coverage
metric); x range. HalfCheetahGoal (METRA App. F.1: 256 goals in [-100, 100] m, 3 m, 200 steps): reach = success at any
step, final = final-step success. Behaviour: flipped = cos(pitch) < 0 at T; spinning = >= 1 full pitch rotation;
stationary = |x_T| <= 5 m; modes = (backward / stationary / forward) x (upright / flipped) with entropy in bits; posture
diversity = mean pairwise L2 distance of time-mean (t >= 100) [cos p, sin p, rootz, 6 joints]; R^2 of linear fits of x_T
and of the time-mean cos pitch from u_T = phi(s_T) - phi(s_0); latent error = mean angle(u_T, z).

    python eval/half_cheetah/summarize.py   -> results/half_cheetah/summary.md, results/half_cheetah/fig_hc.png
"""
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.spatial.distance import pdist

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, 'results', 'half_cheetah')
METHODS = [('dart_adaptive', 'DART (Adaptive)'), ('dart', 'DART (Constant)'), ('metra', 'METRA'),
           ('nm', 'Norm Matching'), ('d3', 'D3 Schedule')]
NAME = dict(METHODS)
TOL = 3.0
wrap = lambda a: (a + 180) % 360 - 180


def seeds_of(sub, suffix):
    out = {}
    for m, _ in METHODS:
        fs = glob.glob(os.path.join(R, sub, f'{m}_s*{suffix}'))
        ss = sorted(int(re.search(r'_s(\d+)' + re.escape(suffix) + '$', f).group(1)) for f in fs)
        if ss:
            out[m] = ss
    return out


def r2(u, y):
    A = np.c_[u, np.ones(len(u))]
    res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - res.var() / max(y.var(), 1e-12)


def protocol_metrics(d):
    q = d['qpos']
    x = q[:, :, 0] - q[:, :1, 0]
    p = q[:, :, 2]
    xT = x[:, -1]
    cp = np.cos(p)
    mcp = cp.mean(1)
    direc = np.where(xT > 5, 2, np.where(xT < -5, 0, 1))
    frac = np.bincount(direc * 2 + (mcp < 0), minlength=6) / len(xT)
    nz = frac[frac > 0]
    feat = np.c_[cp[:, 100:].mean(1), np.sin(p[:, 100:]).mean(1), q[:, 100:, 1:2].mean(1), q[:, 100:, 3:].mean(1)]
    ang = np.degrees(np.arctan2(d['u'][:, 1], d['u'][:, 0]))
    gd = np.abs(d['goal_x'] - d['goal'][:, None])
    return dict(absx=np.abs(xT).mean(), bins=len(np.unique(np.floor(x))), xrange=np.ptp(x),
                h1_reach=(gd <= TOL).any(1).mean(), h1_final=(gd[:, -1] <= TOL).mean(), fdist=gd[:, -1].mean(),
                flip=np.mean(cp[:, -1] < 0), spin=np.mean(np.abs(p[:, -1] - p[:, 0]) >= 2 * np.pi),
                still=np.mean(direc == 1), ent=-(nz * np.log2(nz)).sum(), posture=pdist(feat).mean(),
                r2x=r2(d['u'], xT), r2p=r2(d['u'], mcp), lat=np.abs(wrap(ang - d['z'])).mean())


def pm(v, dec):
    v = np.asarray(v, float)
    return f'{v.mean():.{dec}f}' + (f' ± {v.std(ddof=1):.{dec}f}' if len(v) > 1 else '')


L = ['# HalfCheetah results', '', 'Mean ± s.d. over training seeds (number of seeds in parentheses).', '']
P = {}
for m, ss in seeds_of('protocol', '.npz').items():
    rs = [protocol_metrics(np.load(os.path.join(R, 'protocol', f'{m}_s{s}.npz'))) for s in ss]
    P[m] = {k: [r[k] for r in rs] for k in rs[0]}
for m, ss in seeds_of('goal_ctrl', '_free.npz').items():
    rs = []
    for s in ss:
        d = np.load(os.path.join(R, 'goal_ctrl', f'{m}_s{s}_free.npz'))
        r = {}
        for c in ['h1', 'h25', 'adr', 'adr_nostop']:
            dist = np.abs(d[f'{c}_x'] - d['goal'][:, None])
            r[f'{c}_reach'], r[f'{c}_final'] = (dist <= TOL).any(1).mean(), (dist[:, -1] <= TOL).mean()
        rs.append(r)
    P.setdefault(m, {}).update({k: [r[k] for r in rs] for k in rs[0]})
for m, ss in seeds_of('goal_ctrl', '_sac.npz').items():
    P.setdefault(m, {})['sac_reach'] = [np.load(os.path.join(R, 'goal_ctrl', f'{m}_s{s}_sac.npz'))['sac_curve'][-3:].mean()
                                        for s in ss]

ROWS = [('Exploration', [('absx', 'Mean abs(x_T) (m)', 1), ('bins', 'Visited 1 m x-bins', 0), ('xrange', 'x range (m)', 1)]),
        ('HalfCheetahGoal', [('h1_reach', 'h=1 reach', 2), ('h1_final', 'h=1 final', 2), ('h25_reach', 'h=25 reach', 2),
                             ('h25_final', 'h=25 final', 2), ('adr_reach', 'ADR reach', 2), ('adr_final', 'ADR final', 2),
                             ('adr_nostop_final', 'ADR, no stop, final', 2), ('sac_reach', 'High-level SAC reach', 2),
                             ('fdist', 'Final abs(x - g) (m), h=1', 1)]),
        ('Behaviour', [('lat', 'Latent angular error (°)', 1), ('flip', 'Flipped at T', 2), ('spin', 'Spinning', 2),
                       ('still', 'Stationary (abs(x_T) <= 5 m)', 2), ('ent', 'Mode entropy (bits)', 2),
                       ('posture', 'Posture diversity', 2), ('r2x', 'R^2(x_T given u_T)', 2),
                       ('r2p', 'R^2(mean cos pitch given u_T)', 2)])]
ms = [m for m, _ in METHODS if m in P]
if ms:
    L += ['| metric | ' + ' | '.join(NAME[m] for m in ms) + ' |', '|---' * (len(ms) + 1) + '|']
    for grp, rows in ROWS:
        L.append(f'| *{grp}* |' + ' |' * len(ms))
        for k, name, dec in rows:
            if any(k in P[m] for m in ms):
                L.append(f'| {name} | ' + ' | '.join(pm(P[m][k], dec) if k in P[m] else '–' for m in ms) + ' |')
os.makedirs(R, exist_ok=True)
open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))

# Figure (first seed of each method): top, x(t) of 24 evenly spaced skills coloured by skill angle (dashed = flipped at
# T); bottom, latent endpoints u_T of all skills coloured by x_T (hollow = flipped at T).
S = seeds_of('protocol', '.npz')
show = [(m, n) for m, n in METHODS if m in S]
if show:
    INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.titlesize': 8.5, 'axes.titlecolor': INK, 'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
    DIV = LinearSegmentedColormap.from_list('div', ['#2a6fb0', '#d9d8d2', '#c8642a'])
    NORM = TwoSlopeNorm(0, -110, 110)
    fig, axs = plt.subplots(2, len(show), figsize=(1.4 * len(show), 3.3), squeeze=False,
                            gridspec_kw=dict(hspace=0.75, wspace=0.5))
    t = None
    for j, (m, n) in enumerate(show):
        d = np.load(os.path.join(R, 'protocol', f'{m}_s{S[m][0]}.npz'))
        q = d['qpos']
        x = q[:, :, 0] - q[:, :1, 0]
        t = np.arange(x.shape[1]) * 0.05
        flip = np.cos(q[:, -1, 2]) < 0
        ax = axs[0, j]
        for i in range(0, len(x), max(1, len(x) // 24)):
            ax.plot(t, x[i], color=plt.cm.twilight(d['z'][i] / 360), lw=1.0, ls='--' if flip[i] else '-')
        ax.set_ylim(-120, 120)
        ax.set_title(n)
        ax.set_xlabel('time (s)')
        if j == 0:
            ax.set_ylabel('$x$ (m)')
        else:
            ax.set_yticklabels([])
        ax = axs[1, j]
        u = d['u']
        sc = ax.scatter(u[~flip, 0], u[~flip, 1], c=x[~flip, -1], cmap=DIV, norm=NORM, s=6, lw=0)
        ax.scatter(u[flip, 0], u[flip, 1], facecolors='none', edgecolors=DIV(NORM(x[flip, -1])), s=8, lw=0.6)
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel(r'$u_{T,1}$')
        if j == 0:
            ax.set_ylabel(r'$u_{T,2}$')
        for a, letter in zip(axs[:, j], ['abcde'[j], 'fghij'[j]]):
            a.grid(color=GRID, lw=0.5)
            a.tick_params(labelsize=7)
            a.annotate(f'({letter})', xy=(0.5, 0), xycoords=a.xaxis.label, xytext=(0, -3), textcoords='offset points',
                       ha='center', va='top', fontsize=8.5, color=INK)
    cb = fig.colorbar(sc, ax=axs[1, :], fraction=0.02, pad=0.015)
    cb.set_label('$x_T$ (m)')
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(R, 'fig_hc.png'), dpi=300, bbox_inches='tight')
