#!/usr/bin/env python3
"""Ant result tables and trajectory figure from the evaluation outputs in results/ant/ (mean ± s.d. over seeds).

Labels are <method>_s<seed>, method in METHODS below (e.g. dart_s0). Inputs (any missing part is skipped):
  results/ant/rollouts/<label>.npz        eval/ant/rollout_2000.py      -> directional consistency table + figure
  results/ant/goal_reach/<label>.npz      eval/ant/goal_reach.py        -> goal-reaching table
  results/ant/multigoals_sac/<label>.csv  eval/ant/ant_multigoals_hrl.py -> high-level SAC row (mean of last 3 evals)
  results/ant/randstart/<label>.npz       eval/ant/rollout_randstart.py -> random-start table

Directional consistency (2000 evenly spaced unit z, fixed start, 200 steps):
  latent error   angle(phi(s_T) - phi(s_0), z)
  physical error angle(x-y displacement, Q z), Q the best orthogonal map from skill space to the floor (phi's axes are
                 arbitrary), fitted over all skills (orthogonal Procrustes)
  gap            largest angular gap between the x-y end directions of neighbouring skills
  coverage       MjNumUniqueCoords of the run's own epoch-10000 evaluation (progress_eval.csv, 48 random z)
Goal reaching (moving start, 256 goals 10-40 m, 400 steps, 3 m): final-step and any-step success; AntMultiGoals return.

    python eval/ant/summarize.py      -> results/ant/summary.md, results/ant/fig_ant.png
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, 'results', 'ant')
METHODS = [('dart_adaptive', 'DART (Adaptive)', '#1baf7a'), ('dart', 'DART (Constant)', '#eb6834'),
           ('metra', 'METRA', '#2a78d6'), ('nm', 'Norm Matching', '#7a5bc4'), ('d3', 'D3 Schedule', '#c23b6e')]
NAME = {m: n for m, n, _ in METHODS}
TOL = 3.0
wrap = lambda a: (a + 180) % 360 - 180
ang = lambda v: np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360


def seeds_of(sub, ext):
    out = {}
    for m, *_ in METHODS:
        fs = glob.glob(os.path.join(R, sub, f'{m}_s*.{ext}'))
        ss = sorted(int(re.search(r'_s(\d+)\.' + ext + '$', f).group(1)) for f in fs)
        if ss:
            out[m] = ss
    return out


def largest_gap(a):
    a = np.sort(a)
    gaps = np.diff(np.concatenate([a, a[:1] + 360]))
    i = np.argmax(gaps)
    return gaps[i], a[i]


def coverage(run):
    f = os.path.join(str(run), 'progress_eval.csv')
    if not os.path.exists(f):
        return np.nan
    rows = list(csv.DictReader(open(f)))
    key = next(k for k in rows[0] if k.endswith('MjNumUniqueCoords'))
    return {int(float(r['TotalEpoch'])): float(r[key]) for r in rows if r[key]}.get(10000, np.nan)


def consistency(d):
    z = np.stack([np.cos(np.radians(d['z'])), np.sin(np.radians(d['z']))], 1)
    xy = d['qpos'][:, -1, :2] - d['qpos'][:, 0, :2]
    u = d['phi'][:, -1] - d['phi'][:, 0]
    lat = np.abs(wrap(ang(u) - d['z']))
    x = xy / np.linalg.norm(xy, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(x.T @ z)
    phys = np.abs(wrap(ang(xy) - ang(z @ (U @ Vt).T)))
    gap, gap_at = largest_gap(ang(xy))
    return dict(lat=lat.mean(), lat_p95=np.percentile(lat, 95), phys=phys.mean(), phys_p95=np.percentile(phys, 95),
                gap=gap, dist=np.linalg.norm(xy, axis=1).mean(),
                cov=coverage(d['run']) if 'run' in d.files else np.nan), gap_at


def pm(v, dec):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return '–'
    v = v[~np.isnan(v)]
    return f'{v.mean():.{dec}f}' + (f' ± {v.std(ddof=1):.{dec}f}' if len(v) > 1 else '')


def table(title, cols, rows, note=''):
    L = [f'## {title}', '']
    if note:
        L += [note, '']
    L += ['| method | ' + ' | '.join(c for c, *_ in cols) + ' |', '|---' * (len(cols) + 1) + '|']
    for m, vals in rows:
        L.append(f'| {NAME[m]} ({len(next(iter(vals.values())))}) | '
                 + ' | '.join(pm(vals[k], dec) for _, k, dec in cols) + ' |')
    return L + ['']


def welch_line(vals, a, b, keys):
    if a not in vals or b not in vals:
        return []
    out = []
    for k in keys:
        x, y = np.asarray(vals[a][k], float), np.asarray(vals[b][k], float)
        if len(x) > 1 and len(y) > 1:
            out.append(f'{k} p = {stats.ttest_ind(x, y, equal_var=False).pvalue:.3f}')
    return [f'Welch t-test, {NAME[a]} vs {NAME[b]}: ' + ', '.join(out), ''] if out else []


L = ['# Ant results', '', 'Mean ± s.d. over training seeds (number of seeds in parentheses).', '']

# 1. directional consistency + figure
S = seeds_of('rollouts', 'npz')
CONS, GAP_AT = {}, {}
for m, ss in S.items():
    rs = []
    for s in ss:
        r, at = consistency(np.load(os.path.join(R, 'rollouts', f'{m}_s{s}.npz'), allow_pickle=True))
        rs.append(r)
        GAP_AT[(m, s)] = (r['gap'], at)
    CONS[m] = {k: [r[k] for r in rs] for k in rs[0]}
if CONS:
    cols = [('latent err (°)', 'lat', 1), ('latent err p95 (°)', 'lat_p95', 1), ('physical err (°)', 'phys', 1),
            ('physical err p95 (°)', 'phys_p95', 1), ('largest gap (°)', 'gap', 1), ('distance (m)', 'dist', 1),
            ('coverage (bins)', 'cov', 0)]
    L += table('Directional consistency (2000 evenly spaced skills, fixed start, 200 steps)', cols,
               [(m, CONS[m]) for m, *_ in METHODS if m in CONS])
    L += welch_line(CONS, 'dart', 'metra', ['lat', 'phys', 'gap', 'dist'])

    INK, MUTED = '#0b0b0b', '#898781'
    plt.rcParams.update({'font.size': 11.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
                         'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
    show = [(m, n, c) for m, n, c in METHODS if m in S]
    fig = plt.figure(figsize=(7.0 * len(show) / 3, 2.7))
    gs = fig.add_gridspec(1, len(show), wspace=0.3)
    for i, (m, n, col) in enumerate(show):
        s = S[m][0]
        d = np.load(os.path.join(R, 'rollouts', f'{m}_s{s}.npz'), allow_pickle=True)
        tr = d['qpos'][:, :, :2] - d['qpos'][:, :1, :2]
        ax = fig.add_subplot(gs[0, i])
        for k in range(0, len(tr), 25):
            ax.plot(tr[k, :, 0], tr[k, :, 1], color='#c3c2b7', lw=0.4, zorder=1)
        ax.scatter(tr[:, -1, 0], tr[:, -1, 1], s=1.2, color=col, lw=0, zorder=2)
        gap, at = GAP_AT[(m, s)]
        if gap > 5:  # shade the largest empty direction
            th = np.radians(np.linspace(at, at + gap, 30))
            ax.fill(np.r_[0, 75 * np.cos(th), 0], np.r_[0, 75 * np.sin(th), 0], color=col, alpha=0.12, lw=0, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(-72, 72)
        ax.set_ylim(-72, 72)
        ax.set_xticks([-50, 0, 50])
        ax.set_yticks([-50, 0, 50])
        ax.set_xlabel('$x$ (m)')
        if i == 0:
            ax.set_ylabel('$y$ (m)')
        ax.annotate(f'({"abcde"[i]}) {n}', xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -5),
                    textcoords='offset points', ha='center', va='top', fontsize=12.5, color=INK)
    fig.savefig(os.path.join(R, 'fig_ant.png'), dpi=200, bbox_inches='tight')

# 2. goal reaching
S = seeds_of('goal_reach', 'npz')
GOAL = {}
CTRLS = [('bearing1', 'h=1'), ('bearing25', 'h=25'), ('adr', 'ADR'), ('bearing1_stop', 'h=1 + stop'),
         ('bearing25_stop', 'h=25 + stop'), ('adr_nostop', 'ADR, no stop')]
for m, ss in S.items():
    rs = []
    for s in ss:
        d = np.load(os.path.join(R, 'goal_reach', f'{m}_s{s}.npz'))
        r = {}
        for c, _ in CTRLS:
            if f'{c}_xy' in d.files:
                dist = np.linalg.norm(d[f'{c}_xy'] - d[f'{c}_goals'][:, None], axis=2)
                r[f'{c}_final'], r[f'{c}_reach'] = (dist[:, -1] <= TOL).mean(), (dist <= TOL).any(1).mean()
                r[f'{c}_sel'] = d[f'{c}_selections'].mean()
        r['mg_adr'] = d['mg_adr'].mean() if 'mg_adr' in d.files else np.nan
        f = os.path.join(R, 'multigoals_sac', f'{m}_s{s}.csv')
        r['mg_sac'] = np.mean([float(x['return']) for x in csv.DictReader(open(f))][-3:]) if os.path.exists(f) else np.nan
        rs.append(r)
    keys = set.intersection(*[set(r) for r in rs])
    GOAL[m] = {k: [r[k] for r in rs] for k in keys}
if GOAL:
    have = [c for c in CTRLS if all(f'{c[0]}_final' in GOAL[m] for m in GOAL)]
    cols = [(f'{n} {k}', f'{c}_{k}', 2) for c, n in have for k in ['final', 'reach']]
    cols += [('AntMultiGoals ADR', 'mg_adr', 2), ('AntMultiGoals SAC', 'mg_sac', 2)]
    L += table('Goal reaching', cols, [(m, GOAL[m]) for m, *_ in METHODS if m in GOAL],
               'Moving start, 256 goals 10-40 m, 400 steps, 3 m tolerance; final = final-step success, reach = '
               'success at any step. AntMultiGoals: return (max 10).')
    L += table('Skill selections per episode', [(n, f'{c}_sel', 1) for c, n in have],
               [(m, GOAL[m]) for m, *_ in METHODS if m in GOAL])
    L += welch_line(GOAL, 'dart', 'metra', ['adr_final', 'adr_reach', 'mg_adr'])

# 3. random start
S = seeds_of('randstart', 'npz')
RS = {}
for m, ss in S.items():
    per = []
    for s in ss:
        d = np.load(os.path.join(R, 'randstart', f'{m}_s{s}.npz'))
        row = {}
        for c in ['fixed', 'static', 'moving']:
            z = np.stack([np.cos(np.radians(d[f'{c}_z'])), np.sin(np.radians(d[f'{c}_z']))], 1)
            g, ph, di, fl = [], [], [], []
            for k in range(len(d[f'{c}_xy'])):
                xy = d[f'{c}_xy'][k, :, -1]
                x = xy / np.maximum(np.linalg.norm(xy, axis=1, keepdims=True), 1e-9)
                U, _, Vt = np.linalg.svd(x.T @ z)
                g.append(largest_gap(ang(xy))[0])
                ph.append(np.abs(wrap(ang(xy) - ang(z @ (U @ Vt).T))).mean())
                di.append(np.linalg.norm(xy, axis=1).mean())
                fl.append(np.mean(d[f'{c}_up'][k] < 0))
            row.update({f'{c}_gap': np.mean(g), f'{c}_phys': np.mean(ph), f'{c}_dist': np.mean(di),
                        f'{c}_flip': np.mean(fl)})
        per.append(row)
    RS[m] = {k: [r[k] for r in per] for k in per[0]}
if RS:
    cols = [(f'{c} {k}', f'{c}_{k}', 1 if k != 'flip' else 2) for c in ['fixed', 'static', 'moving']
            for k in ['gap', 'phys', 'dist', 'flip']]
    L += table('Random-start robustness (500 evenly spaced skills, 200 steps)', cols,
               [(m, RS[m]) for m, *_ in METHODS if m in RS],
               'fixed: the fixed start; static: 8 random poses with torso yaw 45k deg, at rest; moving: the same '
               'starts after a 50-step pre-roll with another skill. gap/phys in degrees, dist in m, flip = fraction '
               'of skills ending upside down.')

os.makedirs(R, exist_ok=True)
open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
