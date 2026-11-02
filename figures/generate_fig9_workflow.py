#!/usr/bin/env python3
"""
generate_fig9_workflow.py
=========================
Fig. 9: two-tier EOS monitoring workflow (Main.tex Section 6.1).

The regime is classified from the measured S = alpha/beta of a diagnostic full RLC,
not from S_pred = alpha/beta_ref(alpha), which is a monotonic function of alpha and
carries no SAI information. Between diagnoses the routine tier runs EOS3 with the
held SAI (EOS2 fallback); the re-diagnosis interval follows from the drift cost.

Usage:  python generate_fig9_workflow.py   ->  Fig9_monitoring_workflow.png (same folder)

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = r'$\mu$mol m$^{-2}$ s$^{-1}$'

# fill, edge
C_DIAG = ('#FDE3E1', '#C62828')
C_EST = ('#E3EEFB', '#1565C0')
C_ROUT = ('#E2F2E3', '#2E7D32')
C_OUT = ('#ECEAF6', '#283593')
C_R3 = ('#F2F2F2', '#616161')
C_DEC = ('#FFF6D5', '#E0A100')


def box(ax, x, y, w, h, text, colors, bold_first=True, fs=10.5):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fill, ec=edge, lw=2.0, zorder=2))
    lines = text.split('\n')
    if bold_first and len(lines) > 1:
        ax.text(x, y + h * 0.18, lines[0], ha='center', va='center', fontsize=fs,
                fontweight='bold', zorder=3)
        ax.text(x, y - h * 0.16, '\n'.join(lines[1:]), ha='center', va='center',
                fontsize=fs - 1, zorder=3, linespacing=1.3)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold_first else 'normal', zorder=3, linespacing=1.3)


def diamond(ax, x, y, w, h, text, colors):
    fill, edge = colors
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=fill, ec=edge, lw=2.0, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=10, zorder=3, linespacing=1.3)


def arrow(ax, p, q, color, ls='-', rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=16, color=color,
                                 lw=lw, ls=ls, connectionstyle=f'arc3,rad={rad}', zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0.1, 7.6)
    ax.axis('off')

    # tier headers
    ax.text(2.75, 7.35, 'Diagnostic tier  (start-up, then every $\\Delta t$)', ha='center',
            fontsize=12, fontweight='bold', color=C_DIAG[1])
    ax.text(8.35, 7.35, 'Routine tier  (daily)', ha='center',
            fontsize=12, fontweight='bold', color=C_ROUT[1])
    ax.plot([5.55, 5.55], [2.6, 4.9], color='#BDBDBD', lw=1.2, ls=(0, (4, 3)), zorder=0)

    # diagnostic tier
    box(ax, 2.75, 6.55, 4.3, 0.95, f'Full RLC\n$I$ up to 2000 {UNIT}', C_DIAG)
    box(ax, 2.75, 5.0, 4.3, 0.95,
        'Ph10 fit: $\\alpha$, $P_{\\max}$, $\\beta$\n'
        '$\\mathrm{SAI} = \\log_{10}\\beta - \\log_{10}\\beta_{\\mathrm{ref}}(\\alpha)$', C_EST)
    diamond(ax, 2.2, 3.55, 2.9, 1.4, 'Regime from\nmeasured $S = \\alpha/\\beta$', C_DEC)
    box(ax, 4.6, 3.55, 1.45, 0.95, 'R3\n$S \\leq 3$\noutside EOS', C_R3, fs=10)
    box(ax, 2.75, 1.95, 4.3, 1.0,
        'R1 ($S > 10$) or R2 ($3 < S \\leq 10$)\nhold SAI and regime\nuntil the next diagnosis', C_EST)

    arrow(ax, (2.75, 6.07), (2.75, 5.48), C_DIAG[1])
    arrow(ax, (2.2, 4.52), (2.2, 4.25), C_EST[1])
    arrow(ax, (3.65, 3.55), (3.87, 3.55), C_DEC[1])
    arrow(ax, (2.2, 2.85), (2.2, 2.45), C_DEC[1])

    # routine tier
    box(ax, 8.35, 6.55, 4.3, 0.95, f'Low-irradiance RLC\n$I < 500$ {UNIT}', C_ROUT)
    box(ax, 8.35, 4.6, 4.3, 0.95, 'Track $\\alpha$, $P_{\\max}$\n(low-$I$ fit)', C_ROUT)
    box(ax, 8.35, 1.95, 4.3, 1.0,
        'EOS3: $P(I;\\,\\alpha, P_{\\max}, \\mathrm{SAI}_{\\mathrm{held}})$\n'
        'EOS2 if no valid SAI', C_ROUT)
    arrow(ax, (8.35, 6.07), (8.35, 5.08), C_ROUT[1])
    arrow(ax, (8.35, 4.12), (8.35, 2.45), C_ROUT[1])
    arrow(ax, (4.9, 1.95), (6.2, 1.95), C_EST[1])

    # re-diagnosis trigger: routine -> diagnostic after the interval
    arrow(ax, (6.2, 6.55), (4.9, 6.55), C_DIAG[1], ls='--', lw=1.6)
    ax.text(5.55, 6.15, 'after $\\Delta t$', ha='center', va='top', fontsize=9.5,
            color=C_DIAG[1], style='italic')
    ax.text(5.55, 5.75, '$\\Delta t$ from the\ntolerated NRMSE\n(SAI drift cost,\nSection 6.1)',
            ha='center', va='top', fontsize=8.5, color=C_DIAG[1], style='italic', linespacing=1.25,
            bbox=dict(fc='white', ec='none', pad=1.5), zorder=3)

    # output
    box(ax, 5.5, 0.6, 10.4, 0.7,
        'Output: full PI-curve prediction; NRMSE bounded by the design law', C_OUT,
        bold_first=False, fs=11)
    arrow(ax, (8.35, 1.45), (8.35, 0.96), C_OUT[1])

    fig.tight_layout()
    out = os.path.join(HERE, 'Fig9_monitoring_workflow.png')
    fig.savefig(out, dpi=300, facecolor='white')
    print('saved', out)


if __name__ == '__main__':
    main()
