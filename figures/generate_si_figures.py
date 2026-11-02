#!/usr/bin/env python3
"""
generate_si_figures.py
======================
Supplementary Information figure & table generation for:
"An equation of state for photoinhibition: a forbidden zone, a diagnosable stress index, and environmental bottleneck channels"

Generates:
  - Fig_S1_alpha_beta_scatter.png   : Full α–β parameter landscape
  - Fig_S2_FZ_artifact_tests.png    : Forbidden zone 6-panel artifact tests
  - Fig_S3_SAI_recovery.png         : In silico SAI recovery simulation
  - Table_S2_extended_stats.csv     : Extended EOS statistics (SI Table S2)

Model and RLC fitting protocol are imported from analysis/ph10_fit.py
(Main.tex Eq. 1; Methods: multi-start Levenberg–Marquardt + R² ≥ 0.95).
The random-number stream is consumed in the same order as
analysis/rerun_canonical.py, so every number shown here matches
analysis/output/canonical_tables.md.

Requires:
  - Opt_ParVal_of_piModels.csv (raw_data/csv.zip; Zenodo DOI: 10.5281/zenodo.16748102)

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import os
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'analysis'))

from ph10_fit import GAMMA_0, I_RLC, ph10_gross, ph10_model as _ph10_model, fit_ph10_multistart  # noqa: E402

# ── Reproducibility ──────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Regime colors
C_R1 = '#4477AA'  # blue
C_R2 = '#EE7733'  # orange
C_R3 = '#CC3311'  # red
C_FZ = '#EE3377'  # pink

# ── Constants ────────────────────────────────────────────
FZ_S_C = 1.0               # Forbidden zone centre: gate-overlap point (I_alpha = I_beta)
FZ_HALF_WIDTH_DEX = 0.1465  # Forbidden zone half-width in log10 S
FZ_LO, FZ_HI = FZ_S_C * 10**-FZ_HALF_WIDTH_DEX, FZ_S_C * 10**FZ_HALF_WIDTH_DEX  # ≈ 0.714, 1.401
M_SLOPE = 0.814             # Scaling law slope
M_INTER = -1.355            # Scaling law intercept
SIGMA_SAI_EOS3 = 0.15       # EOS3 benchmark SAI uncertainty


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
def load_data(path=None):
    """Load and filter Ph10/Ph11 datasets (default: raw_data/csv.zip)."""
    if path is None and os.path.exists('piModels.csv'):
        path = 'piModels.csv'
    if path is None:
        with zipfile.ZipFile(os.path.join(REPO, 'raw_data', 'csv.zip')) as z:
            with z.open('Opt_ParVal_of_piModels.csv') as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    ph10 = df[df['Model_piCurve_pkg'] == 'Ph10'].copy()
    ph11 = df[df['Model_piCurve_pkg'] == 'Ph11'].copy()

    # Quality filter (paper Section 2.1)
    for d in [ph10, ph11]:
        mask = (d['alpha'] > 1e-6) & (d['beta'] > 1e-6) & (d['Pmax'] > 0.01)
        d.drop(d[~mask].index, inplace=True)
        d['R'] = d['R'].fillna(0.0)

    # Derived quantities for Ph10
    ph10['S'] = ph10['alpha'] / ph10['beta']
    ph10['log_alpha'] = np.log10(ph10['alpha'])
    ph10['log_beta'] = np.log10(ph10['beta'])
    ph10['log_S'] = np.log10(ph10['S'])
    ph10['beta_pred'] = 10**(M_SLOPE * ph10['log_alpha'] + M_INTER)
    ph10['SAI'] = ph10['log_beta'] - (M_SLOPE * ph10['log_alpha'] + M_INTER)

    # Regime classification
    ph10['regime'] = 'R3'
    ph10.loc[ph10['S'] > 3, 'regime'] = 'R2'
    ph10.loc[ph10['S'] > 10, 'regime'] = 'R1'

    # Ph11 derived
    ph11['S'] = ph11['alpha'] / ph11['beta']
    ph11['log_S'] = np.log10(ph11['S'])

    print(f"Ph10: {len(ph10)} curves  (R1={sum(ph10.regime=='R1')}, R2={sum(ph10.regime=='R2')}, R3={sum(ph10.regime=='R3')})")
    print(f"Ph11: {len(ph11)} curves")

    return ph10, ph11


# ══════════════════════════════════════════════════════════
# Ph10 MODEL (Main.tex Eq. 1, via analysis/ph10_fit.py)
# ══════════════════════════════════════════════════════════
def ph10_model(I, alpha, beta, Pmax, R, gamma=GAMMA_0):
    """Double-tanh Ph10 model: Pmax·tanh(αI/Pmax)·tanh((Pmax/(βI))^γ) − R."""
    return _ph10_model(I, alpha, beta, Pmax, R, gamma)


def fit_ph10(I, P):
    """RLC fit per Methods protocol; return (alpha, beta, Pmax, R) or None if rejected."""
    fit = fit_ph10_multistart(I, P)
    return fit['params'] if fit['accepted'] else None


def fz_count(S):
    return int(((S >= FZ_LO) & (S <= FZ_HI)).sum())


def mad(x):
    x = np.asarray(x, dtype=float)
    return np.median(np.abs(x - np.median(x))) * 1.4826


# ══════════════════════════════════════════════════════════
# FIG S1: FULL α–β SCATTER PLOT
# ══════════════════════════════════════════════════════════
def fig_s1_scatter(ph10, outpath='Fig_S1_alpha_beta_scatter.png'):
    """Full log10(α)–log10(β) scatter with regime coloring."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot by regime (R3 first so R1/R2 overlay)
    for regime, color, label in [('R3', C_R3, 'R3 ($S \\leq 3$)'),
                                  ('R2', C_R2, 'R2 ($3 < S \\leq 10$)'),
                                  ('R1', C_R1, 'R1 ($S > 10$)')]:
        sub = ph10[ph10['regime'] == regime]
        ax.scatter(sub['log_alpha'], sub['log_beta'],
                   c=color, s=12, alpha=0.5, edgecolors='none', label=f'{label} ($n={len(sub)}$)')

    # OLS regression line
    x_range = np.array([ph10['log_alpha'].min() - 0.2, ph10['log_alpha'].max() + 0.2])
    y_ols = M_SLOPE * x_range + M_INTER
    ax.plot(x_range, y_ols, 'k--', lw=1.5, label=f'OLS: $m={M_SLOPE}$, $b={M_INTER}$')

    # ±1σ_SAI envelope
    sigma_sai = ph10['SAI'].std()
    ax.plot(x_range, y_ols + sigma_sai, 'k:', lw=0.8, alpha=0.6)
    ax.plot(x_range, y_ols - sigma_sai, 'k:', lw=0.8, alpha=0.6, label=f'$\\pm 1\\sigma_{{SAI}}$ ($\\sigma={sigma_sai:.3f}$)')

    # Label outliers |SAI| > 0.5
    outliers = ph10[ph10['SAI'].abs() > 0.5]
    for _, row in outliers.iterrows():
        ax.annotate(row['pi_number'], (row['log_alpha'], row['log_beta']),
                    fontsize=5, alpha=0.7, ha='left',
                    xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('$\\log_{10}\\alpha$')
    ax.set_ylabel('$\\log_{10}\\beta$')
    ax.set_title('Full $\\alpha$–$\\beta$ parameter landscape ($N = 1{,}808$)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"✓ {outpath} saved ({len(outliers)} outliers labeled)")


# ══════════════════════════════════════════════════════════
# FIG S2: FORBIDDEN ZONE TESTS (6-panel)
# ══════════════════════════════════════════════════════════
def fig_s2_fz_tests(ph10, ph11, outpath='Fig_S2_FZ_artifact_tests.png'):
    """Six-panel forbidden zone tests (RNG order matches analysis/rerun_canonical.py)."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

    log_S = ph10['log_S'].values
    S_vals = ph10['S'].values
    N = len(ph10)

    # FZ in log10 space
    fz_lo_log = np.log10(FZ_LO)
    fz_hi_log = np.log10(FZ_HI)
    n_fz_obs = fz_count(S_vals)

    # ── Panel (a): Poisson null (uniform log10 S over the FULL observed range) ──
    ax_a = fig.add_subplot(gs[0, 0])
    total_width = log_S.max() - log_S.min()
    lam = N * (fz_hi_log - fz_lo_log) / total_width

    k_vals = np.arange(max(0, int(lam - 4 * np.sqrt(lam))), int(lam + 4 * np.sqrt(lam)) + 1)
    ax_a.bar(k_vals, poisson.pmf(k_vals, lam), color='grey', alpha=0.6, width=1.0)
    ax_a.axvline(n_fz_obs, color=C_FZ, lw=2, ls='--', label=f'Observed = {n_fz_obs}')
    ax_a.axvline(lam, color='k', lw=1, ls=':', label=f'Expected $\\lambda = {lam:.1f}$')
    p_val = poisson.cdf(n_fz_obs, lam)
    ax_a.set_title(f'(a) Poisson null (uniform over full $\\log_{{10}} S$)\n$p = {p_val:.2e}$')
    ax_a.set_xlabel('FZ count')
    ax_a.set_ylabel('Probability')
    ax_a.set_xlim(-5, k_vals.max() + 5)
    ax_a.legend(fontsize=8)

    # ── Panel (b): Bootstrap ──
    ax_b = fig.add_subplot(gs[0, 1])
    B_boot = 10000
    boot_fz_counts = np.array([fz_count(S_vals[rng.choice(N, size=N, replace=True)])
                               for _ in range(B_boot)])

    ax_b.hist(boot_fz_counts, bins=np.arange(-0.5, boot_fz_counts.max() + 1.5),
              color='steelblue', alpha=0.7, density=True)
    ax_b.axvline(n_fz_obs, color=C_FZ, lw=2, ls='--', label=f'Observed = {n_fz_obs}')
    ci99 = np.percentile(boot_fz_counts, [0.5, 99.5])
    ax_b.axvspan(ci99[0], ci99[1], alpha=0.15, color='blue', label=f'99% CI [{ci99[0]:.0f}, {ci99[1]:.0f}]')
    ax_b.set_title('(b) Bootstrap resampling\n($B = 10{,}000$)')
    ax_b.set_xlabel('FZ count')
    ax_b.set_ylabel('Density')
    ax_b.legend(fontsize=8)

    # ── Panel (c): Permutation test ──
    ax_c = fig.add_subplot(gs[0, 2])
    alpha_vals = ph10['alpha'].values
    beta_vals = ph10['beta'].values
    B_perm = 10000
    perm_fz_counts = np.array([fz_count(rng.permutation(alpha_vals) / rng.permutation(beta_vals))
                               for _ in range(B_perm)])
    # β-only shuffle (B = 2,000; main-text variant) — drawn here to keep the RNG stream aligned
    perm_beta = np.array([fz_count(alpha_vals / rng.permutation(beta_vals)) for _ in range(2000)])

    ax_c.hist(perm_fz_counts, bins=np.arange(-0.5, perm_fz_counts.max() + 1.5),
              color='grey', alpha=0.7, density=True)
    ax_c.axvline(n_fz_obs, color=C_FZ, lw=2, ls='--', label=f'Observed = {n_fz_obs}')
    ax_c.axvline(perm_fz_counts.mean(), color='k', ls=':', lw=1,
                 label=f'Null {perm_fz_counts.mean():.1f} ± {perm_fz_counts.std(ddof=1):.1f}')
    ax_c.set_title(f'(c) Permutation ($B = 10{{,}}000$)\n'
                   f'$p < 10^{{-4}}$; $\\beta$-only (2,000): {perm_beta.mean():.1f} ± {perm_beta.std(ddof=1):.1f}')
    ax_c.set_xlabel('FZ count ($\\alpha$–$\\beta$ coupling destroyed)')
    ax_c.set_ylabel('Density')
    ax_c.legend(fontsize=8)

    # ── Panel (d): Fitter retention (synthetic curves placed in the FZ) ──
    ax_d = fig.add_subplot(gs[1, 0])
    N_synth = 1000
    real_pmax = ph10['Pmax'].values
    real_alpha = ph10['alpha'].values
    synth_S_pre, synth_S_post = [], []
    for _ in range(N_synth):
        s_target = rng.uniform(FZ_LO, FZ_HI)
        a = rng.choice(real_alpha)
        pm = rng.choice(real_pmax)
        P_noisy = ph10_model(I_RLC, a, a / s_target, pm, 0.0) * (1 + rng.normal(0, 0.03, size=len(I_RLC)))
        synth_S_pre.append(s_target)
        popt = fit_ph10(I_RLC, P_noisy)
        if popt is not None:
            synth_S_post.append(popt[0] / popt[1])

    synth_S_pre = np.array(synth_S_pre)
    synth_S_post = np.array(synth_S_post)
    in_fz_post = fz_count(synth_S_post)
    retention = in_fz_post / len(synth_S_post)

    bins_d = np.linspace(-1.0, 2.5, 71)
    ax_d.hist(np.log10(synth_S_pre), bins=bins_d, alpha=0.5, color='grey', label=f'Pre-fit (FZ: {N_synth})')
    ax_d.hist(np.log10(np.clip(synth_S_post, 10**-1.0, 10**2.5)), bins=bins_d, alpha=0.5, color=C_R1,
              label=f'Post-fit, accepted (FZ: {in_fz_post}/{len(synth_S_post)})')
    ax_d.axvspan(fz_lo_log, fz_hi_log, alpha=0.2, color=C_FZ)
    ax_d.set_title(f'(d) Fitter retention in FZ\n{in_fz_post}/{len(synth_S_post)} accepted fits '
                   f'= {100 * retention:.0f}%')
    ax_d.set_xlabel('$\\log_{10} S$')
    ax_d.set_ylabel('Count')
    ax_d.legend(fontsize=8)

    # ── Panel (e): Ph10 vs Ph11 ──
    ax_e = fig.add_subplot(gs[1, 1])

    log_s_ph11 = ph11['log_S'].values
    # Low-S region only
    bins_e = np.linspace(-1.0, 1.5, 50)

    ax_e.hist(log_S[S_vals < 10], bins=bins_e, alpha=0.5, color=C_R1,
              label=f'Ph10 ($\\gamma$ fixed)', density=True)
    ax_e.hist(log_s_ph11[ph11['S'].values < 10], bins=bins_e, alpha=0.5, color=C_R2,
              label=f'Ph11 ($\\gamma$ free)', density=True)
    ax_e.axvspan(fz_lo_log, fz_hi_log, alpha=0.2, color=C_FZ, label='FZ')

    n_fz_ph11 = fz_count(ph11['S'].values)
    ax_e.set_title(f'(e) Ph10 vs Ph11\n(FZ: Ph10={n_fz_obs}, Ph11={n_fz_ph11})')
    ax_e.set_xlabel('$\\log_{10} S$')
    ax_e.set_ylabel('Density')
    ax_e.legend(fontsize=8)

    # ── Panel (f): Gap-detection sensitivity (descriptive; no significance claimed) ──
    ax_f = fig.add_subplot(gs[1, 2])

    multipliers = np.linspace(0.5, 2.0, 15)
    log_s_sorted = np.sort(log_S[S_vals < 3])
    base_nbins = 30
    fz_lo_list, fz_hi_list = [], []

    for mult in multipliers:
        hist_counts, bin_edges = np.histogram(log_s_sorted, bins=max(int(base_nbins * mult), 5))

        # Find largest run of nearly empty bins
        max_gap, gap_lo_idx, gap_hi_idx = 0, None, None
        for j in range(len(hist_counts)):
            if hist_counts[j] <= 1:
                gap_end = j
                while gap_end + 1 < len(hist_counts) and hist_counts[gap_end + 1] <= 1:
                    gap_end += 1
                gap_width = bin_edges[gap_end + 1] - bin_edges[j]
                if gap_width > max_gap:
                    max_gap, gap_lo_idx, gap_hi_idx = gap_width, j, gap_end

        if gap_lo_idx is not None:
            fz_lo_list.append(10**bin_edges[gap_lo_idx])
            fz_hi_list.append(10**bin_edges[gap_hi_idx + 1])
        else:
            fz_lo_list.append(np.nan)
            fz_hi_list.append(np.nan)

    ax_f.plot(multipliers, fz_lo_list, 'o-', color=C_R1, ms=4, label='Detected lower edge')
    ax_f.plot(multipliers, fz_hi_list, 's-', color=C_R2, ms=4, label='Detected upper edge')
    ax_f.axhline(FZ_LO, color=C_R1, ls=':', alpha=0.5)
    ax_f.axhline(FZ_HI, color=C_R2, ls=':', alpha=0.5)
    ax_f.set_yscale('log')
    ax_f.set_xlabel('Bin-width multiplier')
    ax_f.set_ylabel('$S$ at detected gap edge')
    ax_f.set_title(f'(f) Largest-gap edges vs bin width\n($S < 3$, $n = {len(log_s_sorted)}$; descriptive)')
    ax_f.legend(loc='upper left', fontsize=7)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"✓ {outpath} saved")
    print(f"  Poisson λ={lam:.1f}, p={p_val:.2e}; bootstrap 99% CI [{ci99[0]:.0f}, {ci99[1]:.0f}]")
    print(f"  Permutation (both, 10k): {perm_fz_counts.mean():.1f} ± {perm_fz_counts.std(ddof=1):.1f}, "
          f"median {np.median(perm_fz_counts):.0f}; β-only (2k): {perm_beta.mean():.1f} ± {perm_beta.std(ddof=1):.1f}")
    print(f"  Fitter retention: {in_fz_post}/{len(synth_S_post)} accepted = {retention:.3f}; Ph11 FZ = {n_fz_ph11}")


# ══════════════════════════════════════════════════════════
# FIG S3: SAI RECOVERY SIMULATION
# ══════════════════════════════════════════════════════════
def fig_s3_sai_recovery(ph10, outpath='Fig_S3_SAI_recovery.png'):
    """In silico SAI recovery from noisy RLC (multi-start LM, R² ≥ 0.95)."""

    N_sim = 800

    # Stratified sample
    r1 = ph10[ph10['regime'] == 'R1']
    r2 = ph10[ph10['regime'] == 'R2']
    n_r1 = min(int(N_sim * len(r1) / (len(r1) + len(r2))), len(r1))
    n_r2 = min(N_sim - n_r1, len(r2))

    sample_r1 = r1.sample(n=n_r1, random_state=RNG_SEED)
    sample_r2 = r2.sample(n=n_r2, random_state=RNG_SEED)
    sample = pd.concat([sample_r1, sample_r2])

    results = []

    for _, row in sample.iterrows():
        a_true = row['alpha']
        b_true = row['beta']
        sai_true = row['SAI']

        # Generate noisy RLC
        P_true = ph10_model(I_RLC, a_true, b_true, row['Pmax'], row['R'])
        P_noisy = P_true * (1.0 + rng.normal(0, 0.03, size=len(I_RLC)))

        # Re-fit (Methods protocol; rejected fits fall back to EOS2 and carry no SAI)
        popt = fit_ph10(I_RLC, P_noisy)
        if popt is not None:
            a_rec, b_rec = popt[0], popt[1]
            sai_rec = np.log10(b_rec) - (M_SLOPE * np.log10(a_rec) + M_INTER)
            results.append({
                'regime': row['regime'],
                'S_true': row['S'],
                'SAI_true': sai_true,
                'SAI_rec': sai_rec,
                'delta_SAI': sai_rec - sai_true,
                'alpha_true': a_true,
                'alpha_rec': a_rec,
                'beta_true': b_true,
                'beta_rec': b_rec,
            })

    res = pd.DataFrame(results)
    print(f"  SAI recovery: {len(res)} / {len(sample)} fits accepted (R² ≥ 0.95)")

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ── (a) ΔSAI histogram by regime ──
    ax_a = fig.add_subplot(gs[0, 0])
    r1_d = res[res['regime'] == 'R1']['delta_SAI']
    r2_d = res[res['regime'] == 'R2']['delta_SAI']

    bins_a = np.linspace(-1.0, 1.0, 60)
    ax_a.hist(r1_d, bins=bins_a, alpha=0.6, color=C_R1, label=f'R1 (SD$={r1_d.std():.3f}$)', density=True)
    ax_a.hist(r2_d, bins=bins_a, alpha=0.6, color=C_R2, label=f'R2 (SD$={r2_d.std():.3f}$)', density=True)
    ax_a.axvline(0, color='k', ls='-', lw=0.5)
    ax_a.axvline(-0.15, color='grey', ls='--', lw=1, alpha=0.7)
    ax_a.axvline(0.15, color='grey', ls='--', lw=1, alpha=0.7, label='$\\pm 0.15$ target')

    mad_r1, mad_r2 = mad(r1_d), mad(r2_d)
    ax_a.set_title(f'(a) $\\Delta$SAI by regime\nMAD·1.48: R1={mad_r1:.3f}, R2={mad_r2:.3f}')
    ax_a.set_xlabel('$\\Delta$SAI = $\\widehat{{SAI}} - SAI_{{true}}$')
    ax_a.set_ylabel('Density')
    ax_a.legend(fontsize=8)

    # ── (b) ΔSAI vs S ──
    ax_b = fig.add_subplot(gs[0, 1])
    for regime, color in [('R1', C_R1), ('R2', C_R2)]:
        sub = res[res['regime'] == regime]
        ax_b.scatter(np.log10(sub['S_true']), sub['delta_SAI'],
                     c=color, s=8, alpha=0.4, edgecolors='none', label=regime)
    ax_b.axhline(0, color='k', ls='-', lw=0.5)
    ax_b.axhline(-0.15, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax_b.axhline(0.15, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax_b.axvline(np.log10(10), color='k', ls=':', lw=0.8, alpha=0.5, label='$S=10$ boundary')
    ax_b.set_xlabel('$\\log_{10} S_{true}$')
    ax_b.set_ylabel('$\\Delta$SAI')
    ax_b.set_title('(b) Recovery error vs gate variable')
    ax_b.legend(fontsize=8)

    # ── (c) Recovered α vs true α ──
    ax_c = fig.add_subplot(gs[1, 0])
    for regime, color in [('R1', C_R1), ('R2', C_R2)]:
        sub = res[res['regime'] == regime]
        ax_c.scatter(np.log10(sub['alpha_true']), np.log10(sub['alpha_rec']),
                     c=color, s=8, alpha=0.4, edgecolors='none', label=regime)
    lim_c = [min(np.log10(res['alpha_true']).min(), np.log10(res['alpha_rec']).min()) - 0.2,
             max(np.log10(res['alpha_true']).max(), np.log10(res['alpha_rec']).max()) + 0.2]
    ax_c.plot(lim_c, lim_c, 'k--', lw=1, alpha=0.5, label='1:1')
    ax_c.set_xlim(lim_c)
    ax_c.set_ylim(lim_c)
    ax_c.set_xlabel('$\\log_{10}\\alpha_{true}$')
    ax_c.set_ylabel('$\\log_{10}\\hat{\\alpha}$')
    ax_c.set_title('(c) $\\alpha$ recovery')
    ax_c.legend(fontsize=8)
    ax_c.set_aspect('equal')

    # ── (d) Recovered β vs true β ──
    ax_d = fig.add_subplot(gs[1, 1])
    for regime, color in [('R1', C_R1), ('R2', C_R2)]:
        sub = res[res['regime'] == regime]
        ax_d.scatter(np.log10(sub['beta_true']), np.log10(sub['beta_rec']),
                     c=color, s=8, alpha=0.4, edgecolors='none', label=regime)
    lim_d = [min(np.log10(res['beta_true']).min(), np.log10(res['beta_rec']).min()) - 0.2,
             max(np.log10(res['beta_true']).max(), np.log10(res['beta_rec']).max()) + 0.2]
    ax_d.plot(lim_d, lim_d, 'k--', lw=1, alpha=0.5, label='1:1')
    ax_d.set_xlim(lim_d)
    ax_d.set_ylim(lim_d)
    ax_d.set_xlabel('$\\log_{10}\\beta_{true}$')
    ax_d.set_ylabel('$\\log_{10}\\hat{\\beta}$')
    ax_d.set_title('(d) $\\beta$ recovery')
    ax_d.legend(fontsize=8)
    ax_d.set_aspect('equal')

    fig.savefig(outpath)
    plt.close(fig)
    print(f"✓ {outpath} saved")
    print(f"  MAD·1.4826: R1={mad_r1:.3f}, R2={mad_r2:.3f}; SD: R1={r1_d.std():.3f}, R2={r2_d.std():.3f}")

    return res


# ══════════════════════════════════════════════════════════
# TABLE S1: EXTENDED EOS STATS (Main.tex Eqs. EOS2/EOS3, P_gross)
# ══════════════════════════════════════════════════════════
I_EVAL = np.linspace(1, 3000, 200)


def compute_eos_accuracy(ph10, sigma_sai=SIGMA_SAI_EOS3):
    """EOS2 (β_pred) and EOS3 (β_pred·10^(SAI+ε), ε ~ N(0, σ_SAI)) R² and NRMSE per curve."""
    eps = np.random.default_rng(RNG_SEED).normal(0, sigma_sai, len(ph10))
    results = []

    for (_, row), e in zip(ph10.iterrows(), eps):
        a, b, pm = row['alpha'], row['beta'], row['Pmax']

        # Reference (offset-free P_gross)
        P_ref = ph10_gross(I_EVAL, a, b, pm)
        P_eos2 = ph10_gross(I_EVAL, a, row['beta_pred'], pm)
        P_eos3 = ph10_gross(I_EVAL, a, row['beta_pred'] * 10**(row['SAI'] + e), pm)

        ss_tot = max(np.sum((P_ref - P_ref.mean())**2), 1e-10)
        p_range = max(P_ref.max() - P_ref.min(), 1e-10)

        results.append({
            'pi_number': row['pi_number'],
            'regime': row['regime'],
            'S': row['S'],
            'R2_EOS2': 1.0 - np.sum((P_ref - P_eos2)**2) / ss_tot,
            'R2_EOS3': 1.0 - np.sum((P_ref - P_eos3)**2) / ss_tot,
            'NRMSE_EOS2': np.sqrt(np.mean((P_ref - P_eos2)**2)) / p_range * 100,
            'NRMSE_EOS3': np.sqrt(np.mean((P_ref - P_eos3)**2)) / p_range * 100,
        })

    # R3 (S ≤ 3) is outside the EOS validity domain and excluded from the benchmark
    df = pd.DataFrame(results)
    return df[df['regime'] != 'R3'].reset_index(drop=True)


def table_s2_extended(ph10, outpath='Table_S2_extended_stats.csv'):
    """Generate extended EOS stats table."""
    print("  Computing EOS accuracy for all curves (this may take a moment)...")
    eos_df = compute_eos_accuracy(ph10)

    stats = {}
    for regime in ['R1', 'R2', 'All']:
        if regime == 'All':
            sub = eos_df
        else:
            sub = eos_df[eos_df['regime'] == regime]

        for metric in ['R2_EOS2', 'R2_EOS3', 'NRMSE_EOS2', 'NRMSE_EOS3']:
            vals = sub[metric].values
            key = f"{regime}_{metric}"
            stats[key] = {
                'N': len(vals),
                'median': np.median(vals),
                'p25': np.percentile(vals, 25),
                'p75': np.percentile(vals, 75),
                'p5': np.percentile(vals, 5),
                'p95': np.percentile(vals, 95),
            }

    # Save as CSV
    rows = []
    for key, v in stats.items():
        rows.append({'metric': key, **v})
    pd.DataFrame(rows).to_csv(outpath, index=False, float_format='%.4f')

    # Print summary
    print(f"\n  Extended EOS Statistics:")
    print(f"  {'':30s} {'Median':>8s} {'P25':>8s} {'P75':>8s} {'P5':>8s} {'P95':>8s}")
    for key, v in stats.items():
        print(f"  {key:30s} {v['median']:8.3f} {v['p25']:8.3f} {v['p75']:8.3f} {v['p5']:8.3f} {v['p95']:8.3f}")

    print(f"\n✓ {outpath} saved")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Supplementary Information Figure Generation")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    ph10, ph11 = load_data()

    # Fig S1
    print("\n[2/5] Generating Fig S1: α–β scatter plot...")
    fig_s1_scatter(ph10)

    # Fig S2
    print("\n[3/5] Generating Fig S2: FZ tests (6 panels)...")
    print("  (Bootstrap: 10k, Permutation: 10k + 2k β-only, Fitter retention: 1k curves)")
    fig_s2_fz_tests(ph10, ph11)

    # Fig S3
    print("\n[4/5] Generating Fig S3: SAI recovery simulation...")
    print("  (800 curves, 3% noise, 12-step RLC, multi-start LM, R² ≥ 0.95)")
    res = fig_s3_sai_recovery(ph10)

    # Table S2
    print("\n[5/5] Generating Table S2: Extended EOS statistics...")
    table_s2_extended(ph10)

    print("\n" + "=" * 60)
    print("All supplementary materials generated successfully!")
    print("=" * 60)
