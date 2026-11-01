#!/usr/bin/env python3
"""
fit_allakhverdiev_kd_kr.py
==========================
Fig. 2 of eos_thermal/main.tex: Arrhenius analysis of D1 damage and repair
in Synechocystis sp. PCC 6803 (Allakhverdiev & Murata 2004; Ueno et al. 2016),
with the Phaeodactylum acute P_max shift (Rehder et al. 2023) for comparison.

Inputs  (raw_data/data/)
  nishiyama_group_repair_temperature.csv   k_d(T), k_r(T), D1 synthesis(T), Ueno recovery(T)
  rehder2023_cross_species_validation.csv  P_max acute shift, 6 -> 15 C (4 samples)

Fits
  k_d : mean, CV (population and sample SD), and an Arrhenius fit (Ea_d)
  k_r : Arrhenius on 18, 25, 34 C (10 C = 0 is excluded: log undefined)
  D1  : Arrhenius on relative synthesis at 18, 25, 34 C
  T_m, delta_T : Eq. 8 fitted to Ueno 2016 recovery at 30-44 C (free amplitude, ln delta_T),
                 Ea_r fixed at the k_r value; the 20 C point is outside the heat-response
                 series (equal to 30 C) and is not used
  Each Arrhenius fit is an unconstrained least-squares fit of ln k = ln k0 - Ea/(R T)
  (linear in its parameters: no initial values and no bounds, so nothing can stick
  to a bound); the nonlinear fit in k-space is reported as a check, started from the
  log-linear solution and also unbounded.

Panels
  (A) k_d vs T, CV       (B) Arrhenius plot of k_r and D1 synthesis (+ Rehder acute P_max)
  (C) crossover irradiance I_cross(T) = k_r(T) / sigma_PSII, with sigma_PSII = k_d / I
      from the lincomycin data (k_d proportional to photon flux)
  (D) closed-form b_thermal(T) (Eq. 9) with T_m and delta_T fitted to the Ueno et al. (2016)
      recovery data (30-44 C; Ea_r fixed at the k_r fit, recovery taken as proportional to k_r)

Usage (from the repository root):
  python soba2/fit_allakhverdiev_kd_kr.py

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eos_thermal_model import R_GAS, T0_K, THERMAL_DEFAULT, b_thermal, k_r_shape, T_opt_from  # noqa: E402

DATA = os.path.join(REPO, 'raw_data', 'data')
OUT = os.path.join(HERE, 'output')
I_DAMAGE = 2000.0   # umol photons m-2 s-1, lincomycin photodamage assay (Allakhverdiev 2004 Table 1)


def read_blocks(path):
    """Split a commented multi-table CSV into DataFrames, one per header line."""
    blocks, cur = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return [pd.read_csv(io.StringIO(''.join(b))) for b in blocks]


def arrhenius(T_C, k):
    """Unconstrained Arrhenius fit. Returns Ea (kJ/mol), its SE, ln k0, and the k-space check."""
    x = 1.0 / (np.asarray(T_C, float) + T0_K)
    y = np.log(np.asarray(k, float))
    A = np.vstack([np.ones_like(x), x]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    n, p = len(x), 2
    s2 = float(res[0]) / (n - p) if len(res) and n > p else np.nan
    cov = s2 * np.linalg.inv(A.T @ A)
    ea, ea_se = -coef[1] * R_GAS, np.sqrt(cov[1, 1]) * R_GAS
    # check: nonlinear fit in k-space, started from the log-linear solution, unbounded
    popt, _ = curve_fit(lambda xx, lk0, e: np.exp(lk0 - e / (R_GAS) * xx), x, np.exp(y),
                        p0=[coef[0], ea], maxfev=20000)
    return {'Ea': ea, 'Ea_se': ea_se, 'lnk0': coef[0], 'Ea_kspace': popt[1], 'n': n}


def fit_denaturation(T_C, rec, Ea_r):
    """Fit A * k_r_shape(T; Ea_r, T_m, delta_T) to recovery data; returns T_m, delta_T and SEs."""
    def f(T, lnA, Tm, ln_dT):
        return np.exp(lnA) * k_r_shape(T, Ea_r, Tm, np.exp(ln_dT)) / k_r_shape(38.0, Ea_r, 42.0, 3.0)
    popt, pcov = curve_fit(f, T_C, rec, p0=[np.log(rec.max()), 42.0, np.log(2.0)], maxfev=20000)
    se = np.sqrt(np.diag(pcov))
    resid = rec - f(T_C, *popt)
    return {'T_m': popt[1], 'T_m_se': se[1], 'dT': np.exp(popt[2]), 'ln_dT_se': se[2],
            'dT_range_1se': [np.exp(popt[2] - se[2]), np.exp(popt[2] + se[2])],
            'lnA': popt[0], 'Ea_r_fixed': Ea_r, 'n': len(T_C),
            'R2': 1 - np.sum(resid**2) / np.sum((rec - rec.mean())**2),
            'scale': np.exp(popt[0]) / k_r_shape(38.0, Ea_r, 42.0, 3.0)}


def main():
    os.makedirs(OUT, exist_ok=True)
    kd_df, kr_df, ueno_df, d1_df, ueno_d1 = read_blocks(os.path.join(DATA, 'nishiyama_group_repair_temperature.csv'))
    rehder = read_blocks(os.path.join(DATA, 'rehder2023_cross_species_validation.csv'))
    acute = rehder[1]   # sample, Pmax_at_6C, Pmax_at_15C, ratio

    # ── k_d ──
    kd = kd_df['kd_min-1'].values
    kd_T = kd_df['temp_C'].values
    kd_mean = kd.mean()
    cv_pop = 100 * kd.std(ddof=0) / kd_mean
    cv_smp = 100 * kd.std(ddof=1) / kd_mean
    fit_kd = arrhenius(kd_T, kd)

    # ── k_r and D1 synthesis (10 C = 0 excluded) ──
    m = kr_df['kr_abs_min-1'].values > 0
    fit_kr = arrhenius(kr_df['temp_C'].values[m], kr_df['kr_abs_min-1'].values[m])
    md = d1_df['D1_synthesis_relative_pct'].values > 0
    fit_d1 = arrhenius(d1_df['temp_C'].values[md], d1_df['D1_synthesis_relative_pct'].values[md])

    # ── Rehder acute P_max shift (two temperatures) ──
    r = acute['Pmax_at_15C'].values / acute['Pmax_at_6C'].values
    ratio_mean, ratio_sd = r.mean(), r.std(ddof=1)
    ea_rehder = R_GAS * np.log(ratio_mean) / (1 / (6 + T0_K) - 1 / (15 + T0_K))

    # ── denaturation parameters from Ueno 2016 (heat-response series 30-44 C) ──
    mu = ueno_df['temp_C'].values >= 30
    fit_den = fit_denaturation(ueno_df['temp_C'].values[mu].astype(float),
                               ueno_df['PSII_recovery_at_120min_pct'].values[mu].astype(float), fit_kr['Ea'])
    # T_opt := maximum of k_r(T) (derived, not fitted)
    syn = dict(Ea_r=fit_kr['Ea'], T_m=fit_den['T_m'], dT=fit_den['dT'])
    T_opt_syn = T_opt_from(**syn)
    T_opt_def = T_opt_from(**THERMAL_DEFAULT)
    fit_den['T_opt_kr_max'] = T_opt_syn

    # ── crossover irradiance ──
    sigma = kd_mean / I_DAMAGE          # min-1 per (umol m-2 s-1)
    icross_obs = kr_df['kr_abs_min-1'].values / sigma

    R = {
        'kd': {'values': kd.tolist(), 'mean': kd_mean, 'sd_pop': kd.std(ddof=0), 'sd_sample': kd.std(ddof=1),
               'cv_pop_pct': cv_pop, 'cv_sample_pct': cv_smp, 'Ea_d': fit_kd['Ea'], 'Ea_d_se': fit_kd['Ea_se']},
        'kr': fit_kr, 'D1': fit_d1,
        'rehder': {'ratio_mean': ratio_mean, 'ratio_sd': ratio_sd, 'Ea': ea_rehder},
        'denaturation_ueno': fit_den,
        'sigma_PSII_min-1_per_umol': sigma,
        'I_cross_obs': dict(zip(kr_df['temp_C'].astype(int).astype(str), icross_obs.round(1).tolist())),
    }
    with open(os.path.join(OUT, 'arrhenius_allakhverdiev.json'), 'w') as fh:
        json.dump(R, fh, indent=2, default=float)

    print(f"k_d: mean {kd_mean:.4f} /min, SD(pop) {kd.std(ddof=0):.4f}, SD(sample) {kd.std(ddof=1):.4f}; "
          f"CV {cv_pop:.2f}% (pop) / {cv_smp:.2f}% (sample); Ea_d = {fit_kd['Ea']:.1f} ± {fit_kd['Ea_se']:.1f} kJ/mol")
    print(f"k_r (18/25/34 C): Ea_r = {fit_kr['Ea']:.1f} ± {fit_kr['Ea_se']:.1f} kJ/mol (k-space check {fit_kr['Ea_kspace']:.1f})")
    print(f"D1  (18/25/34 C): Ea   = {fit_d1['Ea']:.1f} ± {fit_d1['Ea_se']:.1f} kJ/mol (k-space check {fit_d1['Ea_kspace']:.1f})")
    print(f"Rehder acute Pmax ratio {ratio_mean:.3f} ± {ratio_sd:.3f} (n={len(r)}) -> Ea = {ea_rehder:.1f} kJ/mol")
    print(f"Ueno 30-44 C (Ea_r fixed {fit_kr['Ea']:.1f}): T_m = {fit_den['T_m']:.2f} ± {fit_den['T_m_se']:.2f} C, "
          f"delta_T = {fit_den['dT']:.2f} C (1-SE range {fit_den['dT_range_1se'][0]:.2f}-{fit_den['dT_range_1se'][1]:.2f}), "
          f"R2 = {fit_den['R2']:.3f}, n = {fit_den['n']}")
    print(f"T_opt (max of k_r): Synechocystis fit {T_opt_syn:.2f} C; default set {T_opt_def:.2f} C")
    print(f"sigma_PSII = k_d/I = {sigma:.2e} min-1 per umol m-2 s-1; I_cross obs: {R['I_cross_obs']}")

    # ══════════════════ FIGURE ══════════════════
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Fig. 2 — Arrhenius analysis of D1 damage and repair (Synechocystis sp. PCC 6803)',
                 fontsize=14, fontweight='bold')
    C_KD, C_KR, C_D1, C_RE = '#D85A30', '#1D9E75', '#534AB7', '#1E88E5'

    # (A) k_d
    ax = axes[0, 0]
    ax.errorbar(kd_T, kd, yerr=kd_df['kd_SE'].values, fmt='o', color=C_KD, ms=9, capsize=4,
                label='k$_d$ (lincomycin, 2000 µmol m$^{-2}$ s$^{-1}$)')
    ax.axhline(kd_mean, color=C_KD, ls='--', lw=1.5)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('k$_d$ (min$^{-1}$)')
    ax.set_title('(A) Damage rate k$_d$ is temperature-independent')
    ax.text(0.04, 0.08, f'mean {kd_mean:.3f} min$^{{-1}}$\nCV = {cv_pop:.1f}% (population SD; sample SD: {cv_smp:.1f}%)\n'
            f'Arrhenius fit: E$_{{a,d}}$ = {fit_kd["Ea"]:.1f} ± {fit_kd["Ea_se"]:.1f} kJ mol$^{{-1}}$',
            transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', fc='white', ec=C_KD))
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # (B) Arrhenius plot, each series normalised to its highest temperature
    ax = axes[0, 1]
    xg = np.linspace(1 / (36 + T0_K), 1 / (4 + T0_K), 100)
    for fit, T, k, c, lab in [
        (fit_kr, kr_df['temp_C'].values[m], kr_df['kr_abs_min-1'].values[m], C_KR, 'k$_r$ (PSII repair)'),
        (fit_d1, d1_df['temp_C'].values[md], d1_df['D1_synthesis_relative_pct'].values[md], C_D1, 'D1 synthesis'),
    ]:
        k_ref = np.exp(fit['lnk0'] - fit['Ea'] / (R_GAS * (34 + T0_K)))
        ax.plot(1e3 / (T + T0_K), np.log(k / k_ref), 'o', color=c, ms=9)
        ax.plot(1e3 * xg, (fit['lnk0'] - fit['Ea'] / R_GAS * xg) - np.log(k_ref), '-', color=c, lw=2,
                label=f'{lab}: E$_a$ = {fit["Ea"]:.0f} ± {fit["Ea_se"]:.0f} kJ mol$^{{-1}}$')
    x_re = 1e3 / (np.array([6, 15]) + T0_K)
    ax.plot(x_re, [np.log(1 / ratio_mean), 0.0], 's--', color=C_RE, ms=8, lw=1.5,
            label=f'P. tricornutum acute P$_{{max}}$ (outside β channel): E$_a$ = {ea_rehder:.0f} kJ mol$^{{-1}}$')
    ax.plot(1e3 * xg, -THERMAL_DEFAULT['Ea_r'] / R_GAS * (xg - 1 / (34 + T0_K)), ':', color='grey', lw=1.5,
            label='uncalibrated default E$_{a,r}$ = 63 kJ mol$^{-1}$ (slope reference)')
    ax.set_xlabel('1000 / T (K$^{-1}$)')
    ax.set_ylabel('ln(rate / rate at reference T)')
    ax.set_title(f'(B) Arrhenius behaviour of k$_r$ and D1: E$_{{a,r}}$ = {fit_kr["Ea"]:.0f}–{fit_d1["Ea"]:.0f} kJ mol$^{{-1}}$')
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.97, 'k$_r$, D1: 10 °C (rate = 0) excluded from fit\nP. tricornutum normalised to 15 °C',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top', color='grey')

    # (C) crossover irradiance
    ax = axes[1, 0]
    Tg = np.linspace(5, 46, 300)
    kr_arr = np.exp(fit_kr['lnk0'] - fit_kr['Ea'] / (R_GAS * (Tg + T0_K)))
    ax.plot(Tg, kr_arr / sigma, '-', color=C_KR, lw=2, label=f'Arrhenius cold branch (E$_{{a,r}}$ = {fit_kr["Ea"]:.0f})')
    kr_full = kr_arr * k_r_shape(Tg, syn['Ea_r'], syn['T_m'], syn['dT']) / k_r_shape(Tg, syn['Ea_r'], 1e9, 1.0)
    ax.plot(Tg, kr_full / sigma, '--', color=C_KR, lw=1.5,
            label=f'with denaturation term (T$_m$ = {syn["T_m"]:.1f} °C, δ$_T$ = {syn["dT"]:.1f} °C, Ueno fit)')
    ax.plot(kr_df['temp_C'].values, icross_obs, 'o', color=C_KR, ms=9, label='observed k$_r$ / σ$_{PSII}$')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('I$_{cross}$ (µmol photons m$^{-2}$ s$^{-1}$)')
    ax.set_title('(C) Crossover irradiance I$_{cross}$(T) = k$_r$(T) / σ$_{PSII}$')
    ax.text(0.03, 0.97, f'σ$_{{PSII}}$ = k$_d$ / I = {sigma:.1e} min$^{{-1}}$ per µmol m$^{{-2}}$ s$^{{-1}}$\n'
            'damage exceeds repair above the curve', transform=ax.transAxes, fontsize=8.5, va='top')
    ax.legend(fontsize=8, loc='center left')
    ax.grid(alpha=0.3)

    # (D) closed-form b_thermal; T_opt = argmax k_r(T), so b_thermal <= 1 by definition
    ax = axes[1, 1]
    p = THERMAL_DEFAULT
    for pars, T_opt, col, lab in [
            (syn, T_opt_syn, '#993C1D', f'Synechocystis (fit): E$_{{a,r}}$ = {syn["Ea_r"]:.0f}, '
                                        f'T$_m$ = {syn["T_m"]:.1f}, δ$_T$ = {syn["dT"]:.1f}'),
            (p, T_opt_def, '#888888', f'uncalibrated default: E$_{{a,r}}$ = {p["Ea_r"]:g}, '
                                      f'T$_m$ = {p["T_m"]:g}, δ$_T$ = {p["dT"]:g}')]:
        ax.plot(Tg, b_thermal(Tg, **pars), '-', color=col, lw=2.2,
                label=lab + f'  (T$_{{opt}}$ = {T_opt:.1f})')
        ax.axvline(T_opt, color=col, ls=':', lw=1)
    rec = ueno_df['PSII_recovery_at_120min_pct'].values.astype(float)
    norm = fit_den['scale'] * k_r_shape(T_opt_syn, syn['Ea_r'], syn['T_m'], syn['dT'])
    ax.plot(ueno_df['temp_C'].values[mu], rec[mu] / norm, 'D', color='k', ms=7,
            label='Ueno 2016 recovery (30–44 °C, fitted)')
    ax.plot(ueno_df['temp_C'].values[~mu], rec[~mu] / norm, 'D', mfc='white', mec='k', ms=7,
            label='Ueno 2016, 20 °C (not fitted)')
    ax.text(0.02, 0.62, 'T$_{opt}$ := temperature of maximal k$_r$(T)\n(dotted lines), so b$_{thermal}$ ≤ 1',
            transform=ax.transAxes, fontsize=7.5, color='grey', zorder=1)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('b$_{thermal}$(T)')
    ax.set_ylim(0, 1.6)
    ax.set_title('(D) Closed-form b$_{thermal}$(T): cold branch and denaturation ceiling')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig2_arrhenius_kd_kr.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'saved {path}')


if __name__ == '__main__':
    main()
