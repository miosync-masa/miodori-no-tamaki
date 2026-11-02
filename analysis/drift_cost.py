#!/usr/bin/env python3
"""
drift_cost.py
=============
NRMSE cost of a common SAI drift delta (Main.tex Section 6.1, re-diagnosis interval).

The design law NRMSE = k * sigma_SAI (k = 50.6) is the median over curves with a random
SAI error of SD sigma per curve. A drift since the last diagnosis is instead an offset
common to every curve, so its median cost is larger. This script evaluates it on the
same curves and grid as the design law in rerun_canonical.py:
  - least-squares slopes (origin-constrained, |delta| = 0.02-0.30) for +delta and -delta
  - the drift at which the median NRMSE reaches 5% (linear interpolation)

Usage (from the repository root):
  python analysis/drift_cost.py

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import os
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMMA_0 = np.cosh(1.0)**2
M_SLOPE, M_INTER = 0.814, -1.355       # as in rerun_canonical.py
I_EVAL = np.linspace(1, 3000, 200)     # as in rerun_canonical.py
DELTAS = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
TARGET = 5.0                           # % NRMSE


def ph10_gross(I, alpha, beta, Pmax, gamma=GAMMA_0):
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return Pmax * np.tanh(alpha * I / Pmax) * np.tanh((Pmax / (beta * I))**gamma)


def load_ph10():
    with zipfile.ZipFile(os.path.join(REPO, 'raw_data', 'csv.zip')) as z:
        with z.open('Opt_ParVal_of_piModels.csv') as f:
            df = pd.read_csv(f)
    d = df[df['Model_piCurve_pkg'] == 'Ph10']
    d = d[(d['alpha'] > 1e-6) & (d['beta'] > 1e-6) & (d['Pmax'] > 0.01)].copy()
    S = d['alpha'] / d['beta']
    d['regime'] = np.where(S > 10, 'R1', np.where(S > 3, 'R2', 'R3'))
    return d


def main():
    d = load_ph10()
    a, pm, b = (d[c].values[:, None] for c in ['alpha', 'Pmax', 'beta'])
    bp = 10**(M_SLOPE * np.log10(a) + M_INTER)
    sai = np.log10(b) - np.log10(bp)
    Pt = ph10_gross(I_EVAL, a, b, pm)
    reg = d['regime'].values

    def nrmse(delta):
        Pp = ph10_gross(I_EVAL, a, bp * 10**(sai + delta), pm)
        return 100 * np.sqrt(np.mean((Pp - Pt)**2, 1)) / np.maximum(Pt.max(1) - Pt.min(1), 1e-10)

    print(f'N = {len(d)}; design-law k = 50.6 (random error, for comparison)')
    for sign, name in [(1, 'upward (+delta)'), (-1, 'downward (-delta)')]:
        for rname, m in [('All', np.ones(len(d), bool)), ('R1', reg == 'R1'), ('R2', reg == 'R2')]:
            y = np.array([np.median(nrmse(sign * x)[m]) for x in DELTAS])
            k = (DELTAS @ y) / (DELTAS @ DELTAS)
            r2 = 1 - np.sum((y - k * DELTAS)**2) / np.sum((y - y.mean())**2)
            print(f'{name:18s} {rname:3s}  slope {k:5.1f}  r2 {r2:.4f}')
        grid = np.arange(0.040, 0.0801, 0.001)
        med = np.array([np.median(nrmse(sign * x)) for x in grid])
        j = np.argmax(med >= TARGET)
        x5 = grid[j - 1] + (TARGET - med[j - 1]) * (grid[j] - grid[j - 1]) / (med[j] - med[j - 1])
        print(f'{name:18s} All  median NRMSE = {TARGET:.0f}% at |delta| = {x5:.4f}')


if __name__ == '__main__':
    main()
