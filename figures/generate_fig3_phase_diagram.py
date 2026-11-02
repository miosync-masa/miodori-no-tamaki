#!/usr/bin/env python3
"""
generate_fig3_phase_diagram.py
==============================
Fig. 2 of "An equation of state for photoinhibition: a forbidden zone, a diagnosable stress index, and environmental bottleneck channels":
  (A) histogram of log10 S by regime, with the forbidden zone (FZ) window
      centred on the gate-overlap point S_c = 1
  (B) EOS validity summary table

All numbers are read from analysis/output/canonical.json, so run
analysis/rerun_canonical.py first.

Usage (from the repository root):
  python figures/generate_fig3_phase_diagram.py

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from eos_sensor import FZ_LOW, FZ_HIGH, FZ_S_C, FZ_HALF_WIDTH_DEX  # noqa: E402

C_R1, C_R2, C_R3, C_FZ = '#2196F3', '#FF9800', '#E53935', '#F8BBD0'


def load_ph10():
    with zipfile.ZipFile(os.path.join(REPO, 'raw_data', 'csv.zip')) as z:
        df = pd.read_csv(z.open('Opt_ParVal_of_piModels.csv'))
    d = df[df['Model_piCurve_pkg'] == 'Ph10']
    d = d[(d['alpha'] > 1e-6) & (d['beta'] > 1e-6) & (d['Pmax'] > 0.01)]
    return np.log10(d['alpha'].values / d['beta'].values)


def main():
    with open(os.path.join(REPO, 'analysis', 'output', 'canonical.json')) as fh:
        R = json.load(fh)
    log_S = load_ph10()
    N = len(log_S)
    eos, cnt = R['eos'], R['data']['ph10']
    n_fz = R['fz']['poisson']['obs']

    fig = plt.figure(figsize=(17, 7))
    ax = fig.add_axes([0.05, 0.1, 0.60, 0.78])
    ax_t = fig.add_axes([0.68, 0.1, 0.31, 0.78])

    # ── (A) phase diagram ──
    bins = np.linspace(-1.0, 4.0, 81)
    ax.axvspan(np.log10(FZ_LOW), np.log10(FZ_HIGH), color=C_FZ, alpha=0.6, zorder=0)
    for mask, color, label in [(log_S > 1, C_R1, f"R1: Factorized (N={cnt['R1']})"),
                               ((log_S > np.log10(3)) & (log_S <= 1), C_R2, f"R2: Transition (N={cnt['R2']})"),
                               (log_S <= np.log10(3), C_R3, f"R3: Coupled (N={cnt['R3']})")]:
        ax.hist(log_S[mask], bins=bins, color=color, alpha=0.8, edgecolor='white', lw=0.4, label=label)
    ax.axvline(np.log10(3), color='grey', ls=':', lw=1.5)
    ax.axvline(1.0, color='grey', ls='--', lw=1.5)
    ax.axvline(np.log10(FZ_S_C), color='#C2185B', ls='-', lw=1.0)
    ymax = ax.get_ylim()[1]
    ax.text(np.log10(3) - 0.03, ymax * 0.95, 'S=3', ha='right', va='top', fontsize=9, color='grey')
    ax.text(1.03, ymax * 0.95, 'S=10', ha='left', va='top', fontsize=9, color='grey')
    ax.text(np.log10(FZ_S_C), ymax * 0.42,
            f'Forbidden\nZone\n$S_c = {FZ_S_C:g}$\n$\\pm${FZ_HALF_WIDTH_DEX} dex',
            ha='center', va='center', fontsize=10, color='#C2185B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#C2185B'))
    ax.set_xlabel('log$_{10}$(S = α/β)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlim(-1.25, 4.25)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('A — Phase Diagram: Gate Variable S', fontsize=13, fontweight='bold')

    # ── (B) EOS validity table ──
    ax_t.axis('off')
    ax_t.set_title('B — EOS Validity Domain', fontsize=13, fontweight='bold')
    header = ['Regime', 'N (%)', 'EOS2 R²', 'EOS3 R²', 'Status']
    rows = [
        ('R1\nFactorized', cnt['R1'], f"{eos['R1']['EOS2_R2']:.3f}", f"{eos['R1']['EOS3_R2']:.3f}", 'EOS2 ✓', '#E3F2FD'),
        ('R2\nTransition', cnt['R2'], f"{eos['R2']['EOS2_R2']:.3f}", f"{eos['R2']['EOS3_R2']:.3f}", 'EOS3 (SAI) ✓', '#FFF3E0'),
        ('R3\nCoupled', cnt['R3'], '—', '—', 'Outside EOS', '#FFEBEE'),
        ('FZ\n(S_c = 1)', n_fz, '—', '—', 'Depleted', '#FCE4EC'),
    ]
    xs = [0.02, 0.24, 0.43, 0.61, 0.79]
    for x, h in zip(xs, header):
        ax_t.text(x, 0.93, h, fontsize=10, fontweight='bold', va='center')
    ax_t.plot([0, 1], [0.90, 0.90], color='k', lw=1)
    for i, (name, n, r2a, r2b, status, bg) in enumerate(rows):
        y = 0.80 - i * 0.19
        ax_t.add_patch(plt.Rectangle((0, y - 0.08), 1, 0.16, color=bg, zorder=0))
        for x, txt in zip(xs, [name, f'{n}\n({100 * n / N:.1f}%)', r2a, r2b, status]):
            ax_t.text(x, y, txt, fontsize=9.5, va='center')
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.text(0.0, 0.0, 'Medians, offset-free P$_{gross}$; EOS3 at σ$_{SAI}$ = 0.15.',
              fontsize=8.5, color='grey')

    fig.suptitle('Fig. 3 — Photosynthetic Phase Diagram and EOS Validity', fontsize=14, fontweight='bold')
    out = os.path.join(HERE, 'Fig3_phase_diagram.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ {out} saved (FZ {FZ_LOW:.3f}–{FZ_HIGH:.3f}: {n_fz} curves)')


if __name__ == '__main__':
    main()
