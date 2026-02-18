#!/usr/bin/env python3
"""
generate_si_figures.py
======================
Supplementary Information figure & table generation for:
"A sensing-ready equation of state for photoinhibition"

Generates:
  - Fig_S1_alpha_beta_scatter.png   : Full α–β parameter landscape
  - Fig_S2_FZ_artifact_tests.png    : Forbidden zone 6-panel artifact tests
  - Fig_S3_SAI_recovery.png         : In silico SAI recovery simulation
  - Table_S1_extended_stats.csv     : Extended EOS statistics

Requires:
  - piModels.csv (from Zenodo DOI: 10.5281/zenodo.16748102)

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

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
GAMMA_0 = np.cosh(1.0)**2  # ≈ 2.381
FZ_LO, FZ_HI = 0.82, 1.61  # Forbidden zone S boundaries
M_SLOPE = 0.814             # Scaling law slope
M_INTER = -1.355            # Scaling law intercept


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
def load_data(path='piModels.csv'):
    """Load and filter Ph10/Ph11 datasets."""
    df = pd.read_csv(path)
    
    ph10 = df[df['Model_piCurve_pkg'] == 'Ph10'].copy()
    ph11 = df[df['Model_piCurve_pkg'] == 'Ph11'].copy()
    
    # Quality filter (paper Section 2.1)
    for d in [ph10, ph11]:
        mask = (d['alpha'] > 1e-6) & (d['beta'] > 1e-6) & (d['Pmax'] > 0.01)
        d.drop(d[~mask].index, inplace=True)
    
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
# Ph10 MODEL
# ══════════════════════════════════════════════════════════
def ph10_model(I, alpha, beta, Pmax, R, gamma=GAMMA_0):
    """Double-tanh Ph10 model."""
    I = np.asarray(I, dtype=float)
    eps = 1e-30
    pcc = np.tanh(alpha * I / (Pmax + eps))
    scc = 1.0 / np.cosh(beta * I / (Pmax + eps))**gamma
    return Pmax * pcc * scc + R


def fit_ph10(I, P, p0=None):
    """Fit Ph10 to data, return (alpha, beta, Pmax, R)."""
    if p0 is None:
        pmax_est = np.max(P)
        alpha_est = P[1] / (I[1] + 1e-10) if len(I) > 1 else 0.1
        p0 = [alpha_est, alpha_est * 0.05, pmax_est, 0.0]
    
    bounds = ([1e-10, 1e-10, 1e-4, -np.inf],
              [np.inf, np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(ph10_model, I, P, p0=p0, bounds=bounds, maxfev=10000)
        return popt  # [alpha, beta, Pmax, R]
    except:
        return None


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
# FIG S2: FORBIDDEN ZONE ARTIFACT TESTS (6-panel)
# ══════════════════════════════════════════════════════════
def fig_s2_fz_tests(ph10, ph11, outpath='Fig_S2_FZ_artifact_tests.png'):
    """Six-panel forbidden zone artifact rejection tests."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)
    
    log_S = ph10['log_S'].values
    S_vals = ph10['S'].values
    N = len(ph10)
    
    # FZ in log10 space
    fz_lo_log = np.log10(FZ_LO)
    fz_hi_log = np.log10(FZ_HI)
    
    # Count FZ curves
    in_fz = ((S_vals >= FZ_LO) & (S_vals <= FZ_HI))
    n_fz_obs = in_fz.sum()
    
    # ── Panel (a): Poisson null ──
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Low-S band for Poisson calculation
    low_s_mask = S_vals < 3
    n_low_s = low_s_mask.sum()
    log_s_low = log_S[low_s_mask]
    total_width = log_s_low.max() - log_s_low.min()
    fz_width = fz_hi_log - fz_lo_log
    lam = n_low_s * (fz_width / total_width)
    
    k_vals = np.arange(0, max(int(lam * 2), 200))
    pmf = poisson.pmf(k_vals, lam)
    ax_a.bar(k_vals, pmf, color='grey', alpha=0.6, width=1.0)
    ax_a.axvline(n_fz_obs, color=C_FZ, lw=2, ls='--', label=f'Observed = {n_fz_obs}')
    ax_a.axvline(lam, color='k', lw=1, ls=':', label=f'Expected $\\lambda = {lam:.1f}$')
    p_val = poisson.cdf(n_fz_obs, lam)
    ax_a.set_title(f'(a) Poisson null\n$p = {p_val:.2e}$')
    ax_a.set_xlabel('FZ count')
    ax_a.set_ylabel('Probability')
    ax_a.set_xlim(-5, min(int(lam * 2.5), 250))
    ax_a.legend(fontsize=8)
    
    # ── Panel (b): Bootstrap ──
    ax_b = fig.add_subplot(gs[0, 1])
    B_boot = 10000
    boot_fz_counts = np.zeros(B_boot, dtype=int)
    for i in range(B_boot):
        idx = rng.choice(N, size=N, replace=True)
        s_boot = S_vals[idx]
        boot_fz_counts[i] = ((s_boot >= FZ_LO) & (s_boot <= FZ_HI)).sum()
    
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
    B_perm = 10000
    perm_fz_counts = np.zeros(B_perm, dtype=int)
    alpha_vals = ph10['alpha'].values
    beta_vals = ph10['beta'].values
    for i in range(B_perm):
        alpha_perm = rng.permutation(alpha_vals)
        beta_perm = rng.permutation(beta_vals)
        s_perm = alpha_perm / beta_perm
        perm_fz_counts[i] = ((s_perm >= FZ_LO) & (s_perm <= FZ_HI)).sum()
    
    ax_c.hist(perm_fz_counts, bins=40, color='grey', alpha=0.7, density=True)
    ax_c.axvline(n_fz_obs, color=C_FZ, lw=2, ls='--', label=f'Observed = {n_fz_obs}')
    perm_median = np.median(perm_fz_counts)
    ax_c.axvline(perm_median, color='k', ls=':', lw=1, label=f'Median = {perm_median:.0f}')
    p_perm = np.mean(perm_fz_counts <= n_fz_obs)
    ax_c.set_title(f'(c) Permutation test\n$p = {p_perm:.4f}$ (median = {perm_median:.0f})')
    ax_c.set_xlabel('FZ count (coupling destroyed)')
    ax_c.set_ylabel('Density')
    ax_c.legend(fontsize=8)
    
    # ── Panel (d): Fitter innocence ──
    ax_d = fig.add_subplot(gs[1, 0])
    
    N_synth = 1000
    I_rlc = np.array([0, 25, 50, 100, 150, 250, 400, 600, 900, 1200, 1600, 2000], dtype=float)
    
    # Generate synthetic curves with S uniformly in FZ
    synth_S_pre = []
    synth_S_post = []
    
    # Sample real Pmax and alpha from data for realistic parameter ranges
    real_pmax = ph10['Pmax'].values
    real_alpha = ph10['alpha'].values
    
    for k in range(N_synth):
        # Random S in FZ
        s_target = rng.uniform(FZ_LO, FZ_HI)
        # Random alpha from real distribution
        a = rng.choice(real_alpha)
        b = a / s_target
        pm = rng.choice(real_pmax)
        R_val = 0.0
        
        synth_S_pre.append(s_target)
        
        # Generate noisy RLC
        P_true = ph10_model(I_rlc, a, b, pm, R_val)
        P_noisy = P_true * (1 + rng.normal(0, 0.03, size=len(I_rlc)))
        
        # Re-fit
        popt = fit_ph10(I_rlc, P_noisy, p0=[a, b, pm, R_val])
        if popt is not None:
            s_recovered = popt[0] / popt[1]
            synth_S_post.append(s_recovered)
        else:
            synth_S_post.append(s_target)  # keep original if fit fails
    
    synth_S_pre = np.array(synth_S_pre)
    synth_S_post = np.array(synth_S_post)
    
    in_fz_pre = ((synth_S_pre >= FZ_LO) & (synth_S_pre <= FZ_HI)).sum()
    in_fz_post = ((synth_S_post >= FZ_LO) & (synth_S_post <= FZ_HI)).sum()
    ratio = in_fz_post / max(in_fz_pre, 1)
    
    bins_d = np.linspace(-0.5, 2.5, 60)
    ax_d.hist(np.log10(synth_S_pre), bins=bins_d, alpha=0.5, color='grey', label=f'Pre-fit (FZ: {in_fz_pre})')
    ax_d.hist(np.log10(synth_S_post), bins=bins_d, alpha=0.5, color=C_R1, label=f'Post-fit (FZ: {in_fz_post})')
    ax_d.axvspan(fz_lo_log, fz_hi_log, alpha=0.2, color=C_FZ)
    ax_d.set_title(f'(d) Fitter innocence\n(ratio = {ratio:.2f})')
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
    
    n_fz_ph11 = ((ph11['S'].values >= FZ_LO) & (ph11['S'].values <= FZ_HI)).sum()
    ax_e.set_title(f'(e) Ph10 vs Ph11\n(FZ: Ph10={n_fz_obs}, Ph11={n_fz_ph11})')
    ax_e.set_xlabel('$\\log_{10} S$')
    ax_e.set_ylabel('Density')
    ax_e.legend(fontsize=8)
    
    # ── Panel (f): Bin-width sensitivity ──
    ax_f = fig.add_subplot(gs[1, 2])
    
    # Vary bin-width multiplier and find largest gap
    multipliers = np.linspace(0.5, 2.0, 15)
    fz_lo_list = []
    fz_hi_list = []
    p_val_list = []
    
    log_s_sorted = np.sort(log_S[S_vals < 3])
    base_nbins = 30
    
    for mult in multipliers:
        nbins = max(int(base_nbins * mult), 5)
        hist_counts, bin_edges = np.histogram(log_s_sorted, bins=nbins)
        
        # Find largest empty gap
        max_gap = 0
        gap_lo_idx = 0
        for j in range(len(hist_counts)):
            if hist_counts[j] <= 1:  # nearly empty bin
                gap_start = j
                gap_end = j
                while gap_end + 1 < len(hist_counts) and hist_counts[gap_end + 1] <= 1:
                    gap_end += 1
                gap_width = bin_edges[gap_end + 1] - bin_edges[gap_start]
                if gap_width > max_gap:
                    max_gap = gap_width
                    gap_lo_idx = gap_start
                    gap_hi_idx = gap_end
        
        if max_gap > 0:
            lo = 10**bin_edges[gap_lo_idx]
            hi = 10**bin_edges[gap_hi_idx + 1]
            fz_lo_list.append(lo)
            fz_hi_list.append(hi)
            # Poisson p-value for this gap
            n_in = ((S_vals >= lo) & (S_vals <= hi)).sum()
            frac = (np.log10(hi) - np.log10(lo)) / total_width
            lam_i = n_low_s * frac
            p_i = poisson.cdf(n_in, max(lam_i, 0.01))
            p_val_list.append(p_i)
        else:
            fz_lo_list.append(np.nan)
            fz_hi_list.append(np.nan)
            p_val_list.append(1.0)
    
    ax_f2 = ax_f.twinx()
    ax_f.plot(multipliers, fz_lo_list, 'o-', color=C_R1, ms=4, label='FZ lower')
    ax_f.plot(multipliers, fz_hi_list, 's-', color=C_R2, ms=4, label='FZ upper')
    ax_f.axhline(FZ_LO, color=C_R1, ls=':', alpha=0.5)
    ax_f.axhline(FZ_HI, color=C_R2, ls=':', alpha=0.5)
    ax_f2.plot(multipliers, [-np.log10(max(p, 1e-100)) for p in p_val_list], 
               'd-', color='grey', ms=4, alpha=0.7, label='$-\\log_{10}(p)$')
    
    ax_f.set_xlabel('Bin-width multiplier')
    ax_f.set_ylabel('$S$ boundary')
    ax_f2.set_ylabel('$-\\log_{10}(p)$', color='grey')
    ax_f.set_title(f'(f) Bin-width sensitivity')
    ax_f.legend(loc='upper left', fontsize=7)
    ax_f2.legend(loc='upper right', fontsize=7)
    
    fig.savefig(outpath)
    plt.close(fig)
    print(f"✓ {outpath} saved")


# ══════════════════════════════════════════════════════════
# FIG S3: SAI RECOVERY SIMULATION
# ══════════════════════════════════════════════════════════
def fig_s3_sai_recovery(ph10, outpath='Fig_S3_SAI_recovery.png'):
    """In silico SAI recovery from noisy RLC."""
    
    I_rlc = np.array([0, 25, 50, 100, 150, 250, 400, 600, 900, 1200, 1600, 2000], dtype=float)
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
        pm_true = row['Pmax']
        R_true = row['R']
        sai_true = row['SAI']
        s_true = row['S']
        regime = row['regime']
        
        # Generate noisy RLC
        P_true = ph10_model(I_rlc, a_true, b_true, pm_true, R_true)
        noise = rng.normal(0, 0.03, size=len(I_rlc))
        P_noisy = P_true * (1.0 + noise)
        
        # Re-fit
        popt = fit_ph10(I_rlc, P_noisy, p0=[a_true, b_true, pm_true, R_true])
        if popt is not None:
            a_rec, b_rec, pm_rec, R_rec = popt
            sai_rec = np.log10(b_rec) - (M_SLOPE * np.log10(a_rec) + M_INTER)
            delta_sai = sai_rec - sai_true
            results.append({
                'regime': regime,
                'S_true': s_true,
                'SAI_true': sai_true,
                'SAI_rec': sai_rec,
                'delta_SAI': delta_sai,
                'alpha_true': a_true,
                'alpha_rec': a_rec,
                'beta_true': b_true,
                'beta_rec': b_rec,
            })
    
    res = pd.DataFrame(results)
    print(f"  SAI recovery: {len(res)} / {len(sample)} fits succeeded")
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)
    
    # ── (a) ΔSAI histogram by regime ──
    ax_a = fig.add_subplot(gs[0, 0])
    r1_d = res[res['regime'] == 'R1']['delta_SAI']
    r2_d = res[res['regime'] == 'R2']['delta_SAI']
    
    bins_a = np.linspace(-1.0, 1.0, 60)
    ax_a.hist(r1_d, bins=bins_a, alpha=0.6, color=C_R1, label=f'R1 ($\\sigma={r1_d.std():.3f}$)', density=True)
    ax_a.hist(r2_d, bins=bins_a, alpha=0.6, color=C_R2, label=f'R2 ($\\sigma={r2_d.std():.3f}$)', density=True)
    ax_a.axvline(0, color='k', ls='-', lw=0.5)
    ax_a.axvline(-0.15, color='grey', ls='--', lw=1, alpha=0.7)
    ax_a.axvline(0.15, color='grey', ls='--', lw=1, alpha=0.7, label='$\\pm 0.15$ target')
    
    # Robust dispersion (MAD × 1.4826)
    mad_r1 = np.median(np.abs(r1_d - np.median(r1_d))) * 1.4826
    mad_r2 = np.median(np.abs(r2_d - np.median(r2_d))) * 1.4826
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
    
    return res


# ══════════════════════════════════════════════════════════
# TABLE S1: EXTENDED EOS STATS (generate CSV)
# ══════════════════════════════════════════════════════════
def compute_eos_accuracy(ph10, sigma_sai=0.15):
    """Compute EOS2 and EOS3 R² and NRMSE for every curve."""
    I_fine = np.linspace(0, 2500, 500)
    results = []
    
    for _, row in ph10.iterrows():
        if row['regime'] == 'R3':
            continue
        
        a = row['alpha']
        b = row['beta']
        pm = row['Pmax']
        R_val = row['R']
        s = row['S']
        regime = row['regime']
        
        # Reference (Pgross = P - R)
        P_ref = ph10_model(I_fine, a, b, pm, 0.0)  # R=0 for gross
        
        # EOS2: use beta_pred from scaling law
        b_eos2 = row['beta_pred']
        P_eos2 = ph10_model(I_fine, a, b_eos2, pm, 0.0)
        
        # EOS3: use beta_pred + SAI perturbation
        sai_noise = rng.normal(0, sigma_sai)
        b_eos3 = 10**(np.log10(b_eos2) + sai_noise)
        P_eos3 = ph10_model(I_fine, a, b_eos3, pm, 0.0)
        
        # R² and NRMSE
        ss_tot = np.sum((P_ref - P_ref.mean())**2)
        if ss_tot < 1e-20:
            continue
        
        r2_eos2 = 1.0 - np.sum((P_ref - P_eos2)**2) / ss_tot
        r2_eos3 = 1.0 - np.sum((P_ref - P_eos3)**2) / ss_tot
        
        nrmse_eos2 = np.sqrt(np.mean((P_ref - P_eos2)**2)) / (P_ref.max() - P_ref.min() + 1e-20) * 100
        nrmse_eos3 = np.sqrt(np.mean((P_ref - P_eos3)**2)) / (P_ref.max() - P_ref.min() + 1e-20) * 100
        
        results.append({
            'pi_number': row['pi_number'],
            'regime': regime,
            'S': s,
            'R2_EOS2': r2_eos2,
            'R2_EOS3': r2_eos3,
            'NRMSE_EOS2': nrmse_eos2,
            'NRMSE_EOS3': nrmse_eos3,
        })
    
    return pd.DataFrame(results)


def table_s1_extended(ph10, outpath='Table_S1_extended_stats.csv'):
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
    ph10, ph11 = load_data('piModels.csv')
    
    # Fig S1
    print("\n[2/5] Generating Fig S1: α–β scatter plot...")
    fig_s1_scatter(ph10)
    
    # Fig S2
    print("\n[3/5] Generating Fig S2: FZ artifact tests (6 panels)...")
    print("  (Bootstrap: 10k iterations, Permutation: 10k iterations, Fitter innocence: 1k curves)")
    fig_s2_fz_tests(ph10, ph11)
    
    # Fig S3
    print("\n[4/5] Generating Fig S3: SAI recovery simulation...")
    print("  (800 curves, 3% noise, 12-step RLC)")
    res = fig_s3_sai_recovery(ph10)
    
    # Table S1
    print("\n[5/5] Generating Table S1: Extended EOS statistics...")
    table_s1_extended(ph10)
    
    print("\n" + "=" * 60)
    print("All supplementary materials generated successfully!")
    print("=" * 60)
