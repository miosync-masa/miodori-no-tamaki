#!/usr/bin/env python3
"""
generate_fig1_framework.py
==========================
Fig. 1: overview of the framework.
  (A) baseline susceptibility beta_ref(alpha) from the dataset-wide scaling law
  (B) environmental capacity b_env = min(1, b_thermal, b_carbon); the minimum names
      the most constrained channel, and SAI = -log10 b_env
  (C) effective susceptibility beta_eff = beta_ref(alpha) / b_env
  Lower panel: baseline vs stressed Ph10 PI curves (same alpha, Pmax; b_env = 0.3),
  computed from the model, not drawn by hand.

Usage:  python generate_fig1_framework.py   ->  Fig1_framework.png (same folder)

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA_0 = np.cosh(1.0)**2
SLOPE, INTER = 0.8135, -1.3552        # Main.tex Eq. (scaling law)

# example curve for the lower panel
ALPHA, PMAX, B_ENV = 0.05, 10.0, 0.3

C_BASE = '#1565C0'
C_STRESS = '#757575'
C_THERM = '#D84315'
C_CARB = '#2E7D32'
C_PANEL = '#F4F5F7'


def ph10_gross(I, alpha, beta, Pmax, gamma=GAMMA_0):
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return Pmax * np.tanh(alpha * I / Pmax) * np.tanh((Pmax / (beta * I))**gamma)


def panel(ax, x0, w, label, title):
    ax.add_patch(FancyBboxPatch((x0, 3.05), w, 3.55, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fc=C_PANEL, ec='none', zorder=0))
    ax.text(x0 + 0.15, 6.38, label, fontsize=13, fontweight='bold', va='center')
    ax.text(x0 + 0.62, 6.38, title, fontsize=12, fontweight='bold', va='center')


def box(ax, x, y, w, h, text, ec, fs=12, fc='white', lw=1.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=3, linespacing=1.4)


def arrow(ax, p, q, color='black', lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, zorder=1))


def main():
    fig = plt.figure(figsize=(13, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    ax.axis('off')

    # (A) baseline susceptibility
    panel(ax, 0.2, 3.9, '(A)', 'Baseline susceptibility')
    box(ax, 0.75, 4.75, 0.8, 0.7, r'$\alpha$', C_BASE, fs=15)
    box(ax, 2.7, 4.75, 2.5, 1.15,
        'scaling law\n' r'$\log_{10}\beta_{\mathrm{ref}} = 0.8135\,\log_{10}\alpha$' '\n' r'$-\ 1.3552$',
        C_BASE, fs=10.5)
    arrow(ax, (1.15, 4.75), (1.45, 4.75), C_BASE)
    ax.text(2.15, 3.55, r'light-use efficiency $\alpha$ fixes the' '\n'
            r'baseline susceptibility $\beta_{\mathrm{ref}}(\alpha)$',
            ha='center', va='center', fontsize=10.5, color='#424242', linespacing=1.4)

    # (B) environmental capacity
    panel(ax, 4.35, 4.3, '(B)', 'Environmental capacity')
    box(ax, 5.45, 5.55, 1.85, 0.75, r'$b_{\mathrm{thermal}}(T)$', C_THERM, fs=12)
    box(ax, 7.55, 5.55, 2.05, 0.75, r'$b_{\mathrm{carbon}}(\mathrm{pH}, \mathrm{DIC}, T)$', C_CARB, fs=11)
    box(ax, 6.5, 4.3, 3.6, 0.8, r'$b_{\mathrm{env}} = \min(1,\ b_{\mathrm{thermal}},\ b_{\mathrm{carbon}})$',
        'black', fs=12.5, lw=2.2)
    arrow(ax, (5.6, 5.17), (6.0, 4.72), C_THERM)
    arrow(ax, (7.4, 5.17), (7.0, 4.72), C_CARB)
    ax.text(6.5, 3.45, 'the minimum names the most constrained channel;\n'
            r'$\mathrm{SAI} = -\log_{10} b_{\mathrm{env}}$',
            ha='center', va='center', fontsize=10.5, color='#424242', linespacing=1.4)

    # (C) effective susceptibility
    panel(ax, 8.85, 3.95, '(C)', 'Effective susceptibility')
    box(ax, 10.82, 4.75, 2.6, 1.05,
        r'$\beta_{\mathrm{eff}} = \dfrac{\beta_{\mathrm{ref}}(\alpha)}{b_{\mathrm{env}}}$', 'black', fs=15)
    ax.text(10.82, 3.5, r'$b_{\mathrm{env}} = 1$: baseline, $\beta_{\mathrm{eff}} = \beta_{\mathrm{ref}}$' '\n'
            r'$b_{\mathrm{env}} < 1$: repair limited, $\beta_{\mathrm{eff}} > \beta_{\mathrm{ref}}$',
            ha='center', va='center', fontsize=10.5, color='#424242', linespacing=1.5)

    # flow between panels
    arrow(ax, (3.95, 4.6), (4.68, 4.35), C_BASE)
    ax.text(4.33, 4.68, r'$\beta_{\mathrm{ref}}(\alpha)$', ha='center', fontsize=11, color=C_BASE)
    arrow(ax, (8.32, 4.3), (9.48, 4.6))

    # lower panel: model PI curves
    beta_ref = 10**(SLOPE * np.log10(ALPHA) + INTER)
    I = np.linspace(1, 3000, 600)
    p_base = ph10_gross(I, ALPHA, beta_ref, PMAX)
    p_str = ph10_gross(I, ALPHA, beta_ref / B_ENV, PMAX)
    axp = fig.add_axes([0.33, 0.07, 0.34, 0.31])
    axp.plot(I, p_base, color=C_BASE, lw=2.4, label=r'baseline ($b_{\mathrm{env}} = 1$)')
    axp.plot(I, p_str, color=C_STRESS, lw=2.4, ls='--',
             label=rf'stressed ($b_{{\mathrm{{env}}}} = {B_ENV}$)')
    for beta, c in [(beta_ref, C_BASE), (beta_ref / B_ENV, C_STRESS)]:
        axp.axvline(PMAX / beta, color=c, lw=1.0, ls=':')
    axp.text(PMAX / beta_ref + 40, 10.6, r'$I_\beta$', color=C_BASE, ha='left', fontsize=10)
    axp.text(PMAX * B_ENV / beta_ref + 40, 10.6, r'$I_{\beta,\mathrm{eff}}$', color=C_STRESS, ha='left', fontsize=10)
    axp.set_xlim(0, 3000)
    axp.set_ylim(0, 11.5)
    axp.set_xlabel(r'Irradiance $I$ ($\mu$mol photons m$^{-2}$ s$^{-1}$)', fontsize=10.5)
    axp.set_ylabel(r'$P_{\mathrm{gross}}$', fontsize=11)
    axp.tick_params(labelsize=9)
    axp.spines[['top', 'right']].set_visible(False)
    axp.legend(fontsize=9.5, frameon=False, loc='center', bbox_to_anchor=(0.6, 0.55))
    axp.set_title(rf'Ph10 model, $\alpha = {ALPHA}$, $P_{{\max}} = {PMAX:g}$; '
                  r'reduced capacity moves photoinhibition to lower $I$', fontsize=10)

    out = os.path.join(HERE, 'Fig1_framework.png')
    fig.savefig(out, dpi=300, facecolor='white')
    print('saved', out)


if __name__ == '__main__':
    main()
