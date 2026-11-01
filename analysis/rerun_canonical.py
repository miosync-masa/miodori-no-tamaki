#!/usr/bin/env python3
"""
rerun_canonical.py
==================
End-to-end re-run (seed = 42) of every quantitative result in
"A sensing-ready equation of state for photoinhibition" (PI_Curve Main + SI),
using the manuscript model (Main.tex Eq. 1) and the RLC fitting protocol of
Methods l.319-323 (multi-start Levenberg-Marquardt + R^2 >= 0.95 filter),
implemented in analysis/ph10_fit.py.

Outputs (analysis/output/):
  canonical.json        : all canonical numbers
  canonical_tables.md   : canonical tables + diff against the previous canonical
  sai_recovery.csv      : per-curve SAI recovery (all fit variants)

Usage (from the repository root):
  python analysis/rerun_canonical.py

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from ph10_fit import (GAMMA_0, I_RLC, R2_MIN, ph10_gross, ph10_model,  # noqa: E402
                      fit_ph10_multistart)
from eos_sensor import EOSSensor, W_ALPHA, W_BETA                   # noqa: E402

# ── Reproducibility ──────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Constants (published) ────────────────────────────────
M_SLOPE, M_INTER = 0.814, -1.355     # Main.tex Eq. scaling_law
FZ_S_C = 1.0                         # forbidden zone centre: gate overlap, I_alpha = I_beta
FZ_HALF_WIDTH_DEX = 0.1465           # = 0.5 * log10(1.61 / 0.82), width of the earlier empirical gap
FZ_LO, FZ_HI = FZ_S_C * 10**-FZ_HALF_WIDTH_DEX, FZ_S_C * 10**FZ_HALF_WIDTH_DEX   # ≈ 0.714, 1.401
K_SEEDS = 100                        # noise realisations averaged for the design-law slope k
NOISE_CV = 0.03
N_SIM = 800
SIGMA_SAI_EOS3 = 0.15
OUTLIER_DSAI = 0.5                   # gross recovery failure: |dSAI| > 0.5

OUT = os.path.join(HERE, 'output')

# ── Previous canonical (round 2: data-drawn FZ window 0.82 <= S <= 1.61, single-seed k) ──
OLD = {
    'FZ_lo': 0.82, 'FZ_hi': 1.61,
    'N_ph10': 1808, 'N_R1': 1221, 'N_R2': 571, 'N_R3': 16, 'N_ph11': 1805,
    'slope': 0.8135, 'intercept': -1.3552, 'r2_scaling': 0.4274, 'se_slope': 0.0222,
    'sd_SAI_dataset': 0.3031,
    'FZ_obs': 2, 'FZ_ph11': 37,
    'poisson_full_lambda': 110.5, 'poisson_full_p': 6.55e-45,
    'boot_ci99': [0, 7],
    'perm10k_both_mean': 37.0, 'perm10k_both_sd': 5.3, 'perm10k_both_median': 37, 'perm10k_both_Z': -6.6,
    'perm2k_beta_mean': 37.0, 'perm2k_beta_sd': 5.2,
    'fitter_accepted': 989, 'fitter_post': 370, 'fitter_ratio': 0.374,
    'retadj_mean': 13.9, 'retadj_sd': 3.6, 'retadj_p': 2e-4,
    'SAI_acc_R1': 545, 'SAI_acc_R2': 255,
    'SAI_MAD_R1': 0.033, 'SAI_MAD_R2': 0.025, 'SAI_SD_R1': 0.487, 'SAI_SD_R2': 0.260,
    'SAI_cov_R1': 90.6, 'SAI_cov_R2': 98.8, 'SAI_out_R1': 5.7, 'SAI_out_R2': 0.4,
    'eb_MAD_la': 0.024, 'eb_MAD_lb': 0.014, 'eb_SD_la': 0.521, 'eb_SD_lb': 0.058,
    'EOS2_R2_R1': 0.929, 'EOS2_NRMSE_R1': 8.19, 'EOS3_R2_R1': 0.939, 'EOS3_NRMSE_R1': 7.58,
    'EOS2_R2_R2': 0.575, 'EOS2_NRMSE_R2': 18.60, 'EOS3_R2_R2': 0.934, 'EOS3_NRMSE_R2': 7.17,
    'EOS2_R2_All': 0.849, 'EOS2_NRMSE_All': 11.79, 'EOS3_R2_All': 0.937, 'EOS3_NRMSE_All': 7.46,
    'k_design': 48.8, 'k_r2': 0.9991, 'peak10_R1': 100.0,
}


def mad(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))) * 1.4826) if len(x) else np.nan


def fz_count(S):
    return int(((S >= FZ_LO) & (S <= FZ_HI)).sum())


# ══════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════
def load_data():
    with zipfile.ZipFile(os.path.join(REPO, 'raw_data', 'csv.zip')) as z:
        with z.open('Opt_ParVal_of_piModels.csv') as f:
            df = pd.read_csv(f)
    out = {}
    for m in ['Ph10', 'Ph11']:
        d = df[df['Model_piCurve_pkg'] == m].copy()
        out[f'{m}_raw'] = len(d)
        d = d[(d['alpha'] > 1e-6) & (d['beta'] > 1e-6) & (d['Pmax'] > 0.01)].copy()
        d['R'] = d['R'].fillna(0.0)
        d['S'] = d['alpha'] / d['beta']
        d['regime'] = np.where(d['S'] > 10, 'R1', np.where(d['S'] > 3, 'R2', 'R3'))
        out[m] = d
    ph10 = out['Ph10']
    ph10['log_alpha'] = np.log10(ph10['alpha'])
    ph10['log_beta'] = np.log10(ph10['beta'])
    ph10['log_S'] = np.log10(ph10['S'])
    ph10['beta_pred'] = 10**(M_SLOPE * ph10['log_alpha'] + M_INTER)
    ph10['SAI'] = ph10['log_beta'] - (M_SLOPE * ph10['log_alpha'] + M_INTER)
    return out


def counts(d):
    vc = d['regime'].value_counts()
    return {'N': len(d), 'R1': int(vc.get('R1', 0)), 'R2': int(vc.get('R2', 0)),
            'R3': int(vc.get('R3', 0))}


def scaling_law(d):
    lr = stats.linregress(d['log_alpha'], d['log_beta'])
    sai = d['log_beta'] - (lr.slope * d['log_alpha'] + lr.intercept)
    return {'slope': lr.slope, 'intercept': lr.intercept, 'r2': lr.rvalue**2,
            'se_slope': lr.stderr, 'sd_SAI': float(sai.std(ddof=1)), 'N': len(d)}


# ══════════════════════════════════════════════════════════
# 2. FORBIDDEN ZONE: six tests
# ══════════════════════════════════════════════════════════
def poisson_tests(ph10):
    S, logS, N = ph10['S'].values, ph10['log_S'].values, len(ph10)
    n_obs = fz_count(S)
    w_fz = np.log10(FZ_HI) - np.log10(FZ_LO)
    lam_full = N * w_fz / (logS.max() - logS.min())
    return {'obs': n_obs, 'obs_S': sorted(float(x) for x in S[(S >= FZ_LO) & (S <= FZ_HI)]),
            'full_range': {'lambda': lam_full, 'p': stats.poisson.cdf(n_obs, lam_full)}}


def bootstrap_test(ph10, B=10000):
    S = ph10['S'].values
    c = np.array([fz_count(S[rng.choice(len(S), len(S), replace=True)]) for _ in range(B)])
    return {'B': B, 'ci99': [float(np.percentile(c, 0.5)), float(np.percentile(c, 99.5))],
            'median': float(np.median(c))}


PERM_COUNTS = {}


def permutation_test(a, b, B, mode, key=None):
    n_obs = fz_count(a / b)
    c = np.empty(B, dtype=int)
    for i in range(B):
        if mode == 'both':
            c[i] = fz_count(rng.permutation(a) / rng.permutation(b))
        else:
            c[i] = fz_count(a / rng.permutation(b))
    sd = c.std(ddof=1)
    if key is not None:
        PERM_COUNTS[key] = c
    return {'B': B, 'mode': mode, 'obs': n_obs, 'mean': float(c.mean()), 'sd': float(sd),
            'median': float(np.median(c)),
            'iqr': [float(np.percentile(c, 25)), float(np.percentile(c, 75))],
            'Z': float((n_obs - c.mean()) / sd), 'p_le_obs': float(np.mean(c <= n_obs))}


def fitter_innocence(ph10, N_synth=1000):
    real_alpha, real_pmax = ph10['alpha'].values, ph10['Pmax'].values
    rows = []
    for _ in range(N_synth):
        s_t = rng.uniform(FZ_LO, FZ_HI)
        a = rng.choice(real_alpha)
        pm = rng.choice(real_pmax)
        P_noisy = ph10_model(I_RLC, a, a / s_t, pm, 0.0) * (1 + rng.normal(0, NOISE_CV, len(I_RLC)))
        fit = fit_ph10_multistart(I_RLC, P_noisy)
        s_r = fit['params'][0] / fit['params'][1] if fit['params'] is not None else np.nan
        leg = _fit_legacy_trf(I_RLC, P_noisy, [a, a / s_t, pm, 0.0])
        rows.append({'S_true': s_t, 'S_rec': s_r, 'r2': fit['r2'], 'accepted': fit['accepted'],
                     'S_legacy': leg[0] / leg[1] if leg is not None else s_t,
                     'Ialpha': pm / a})
    r = pd.DataFrame(rows)
    acc = r[r['accepted']]
    post = fz_count(acc['S_rec'].values)
    ident = acc[acc['Ialpha'] >= I_RLC[1]]   # PCC saturation at or above the first actinic step
    return {'N_synth': N_synth, 'pre_in_FZ': N_synth, 'converged': int(r['S_rec'].notna().sum()),
            'accepted': len(acc), 'post_in_FZ': post, 'ratio': post / max(len(acc), 1),
            'post_below_FZ': int((acc['S_rec'] < FZ_LO).sum()),
            'post_above_FZ': int((acc['S_rec'] > FZ_HI).sum()),
            'legacy_post_in_FZ': fz_count(r['S_legacy'].values),
            'legacy_ratio': fz_count(r['S_legacy'].values) / N_synth,
            'identifiable_n': len(ident),
            'identifiable_ratio': fz_count(ident['S_rec'].values) / max(len(ident), 1)}


def retention_adjusted_null(perm_counts, n_obs, retention):
    """Apply the measured fitter retention binomially to each permutation-null count."""
    g = np.random.default_rng(RNG_SEED)       # separate stream: does not shift the main RNG
    k = g.binomial(perm_counts, retention)
    return {'retention': retention, 'mean': float(k.mean()), 'sd': float(k.std(ddof=1)),
            'p_le_obs': float(np.mean(k <= n_obs)),
            'retention_needed': n_obs / float(np.mean(perm_counts))}


def _largest_gap(log_s, nbins):
    h, e = np.histogram(log_s, bins=nbins)
    best, lo, hi, j = 0.0, None, None, 0
    while j < len(h):
        if h[j] <= 1:
            k = j
            while k + 1 < len(h) and h[k + 1] <= 1:
                k += 1
            if e[k + 1] - e[j] > best:
                best, lo, hi = e[k + 1] - e[j], e[j], e[k + 1]
            j = k + 1
        else:
            j += 1
    return lo, hi


def binwidth_test(ph10):
    """Descriptive: largest-gap edges in the S < 3 band vs histogram bin width."""
    S, logS = ph10['S'].values, ph10['log_S'].values
    data = np.sort(logS[S < 3])
    lo_l, hi_l = [], []
    for mult in np.linspace(0.5, 2.0, 15):
        lo, hi = _largest_gap(data, max(int(30 * mult), 5))
        if lo is not None:
            lo_l.append(10**lo); hi_l.append(10**hi)
    return {'n_S<3': len(data), 'lower_range': [min(lo_l), max(lo_l)],
            'upper_range': [min(hi_l), max(hi_l)],
            'max_shift_log10': float(max(np.ptp(np.log10(lo_l)), np.ptp(np.log10(hi_l))))}


# ══════════════════════════════════════════════════════════
# 3. SAI RECOVERY (in silico RLC)
# ══════════════════════════════════════════════════════════
def _fit_legacy_trf(I, P, p0):
    """Previous procedure: single start at the TRUE parameters, TRF with bounds, no filter."""
    try:
        popt, _ = curve_fit(ph10_model, I, P, p0=p0,
                            bounds=([1e-10, 1e-10, 1e-4, -np.inf], [np.inf] * 4), maxfev=10000)
        return tuple(popt)
    except (RuntimeError, ValueError):
        return None


def sai_recovery(ph10):
    r1, r2 = ph10[ph10['regime'] == 'R1'], ph10[ph10['regime'] == 'R2']
    n_r1 = min(int(N_SIM * len(r1) / (len(r1) + len(r2))), len(r1))
    n_r2 = min(N_SIM - n_r1, len(r2))
    sample = pd.concat([r1.sample(n=n_r1, random_state=RNG_SEED),
                        r2.sample(n=n_r2, random_state=RNG_SEED)])
    rows = []
    for _, c in sample.iterrows():
        P_true = ph10_model(I_RLC, c['alpha'], c['beta'], c['Pmax'], c['R'])
        P_noisy = P_true * (1.0 + rng.normal(0, NOISE_CV, len(I_RLC)))   # one noise draw, all variants
        base = {'pi_number': c['pi_number'], 'regime': c['regime'], 'S_true': c['S'],
                'SAI_true': c['SAI'], 'alpha_true': c['alpha'], 'beta_true': c['beta']}
        fits = {'legacy': _fit_legacy_trf(I_RLC, P_noisy, [c['alpha'], c['beta'], c['Pmax'], c['R']])}
        ms = fit_ph10_multistart(I_RLC, P_noisy)
        fits['multistart'] = ms['params']
        base['r2_fit'] = ms['r2']
        base['accepted'] = ms['accepted']
        for k, p in fits.items():
            if p is None:
                base[f'{k}_ok'] = False
                continue
            a, b = p[0], p[1]
            base[f'{k}_ok'] = True
            base[f'{k}_alpha'] = a
            base[f'{k}_beta'] = b
            base[f'{k}_dSAI'] = (np.log10(b) - (M_SLOPE * np.log10(a) + M_INTER)) - c['SAI']
        rows.append(base)
    return pd.DataFrame(rows)


def sai_summary(res, variant, use_filter):
    out = {}
    for reg in ['R1', 'R2', 'All']:
        d = res if reg == 'All' else res[res['regime'] == reg]
        n = len(d)
        ok = d[d[f'{variant}_ok']]
        if use_filter:
            ok = ok[ok['accepted']]
        ds = ok[f'{variant}_dSAI']
        la = np.log10(ok[f'{variant}_alpha'] / ok['alpha_true'])
        lb = np.log10(ok[f'{variant}_beta'] / ok['beta_true'])
        out[reg] = {'n': n, 'n_acc': len(ok), 'acc_pct': 100 * len(ok) / max(n, 1),
                    'MAD': mad(ds), 'SD': float(ds.std(ddof=1)),
                    'outlier_pct': 100 * float(np.mean(np.abs(ds) > OUTLIER_DSAI)),
                    'cov015_pct': 100 * float(np.mean(np.abs(ds) < 0.15)),
                    'cov015_of_sampled_pct': 100 * float(np.sum(np.abs(ds) < 0.15)) / max(n, 1),
                    'MAD_la': mad(la), 'MAD_lb': mad(lb),
                    'SD_la': float(la.std(ddof=1)), 'SD_lb': float(lb.std(ddof=1)),
                    'pred_sigma_SAI_MAD': float(np.sqrt(mad(lb)**2 + M_SLOPE**2 * mad(la)**2))}
    return out


# ══════════════════════════════════════════════════════════
# 4. EOS ACCURACY + DESIGN LAW (Main definitions, P_gross)
# ══════════════════════════════════════════════════════════
I_EVAL = np.linspace(1, 3000, 200)


def _metrics(ph10, sai_used):
    a, pm, b = (ph10[c].values[:, None] for c in ['alpha', 'Pmax', 'beta'])
    bp = ph10['beta_pred'].values[:, None]
    Pt = ph10_gross(I_EVAL, a, b, pm)
    Pp = ph10_gross(I_EVAL, a, bp * 10**np.asarray(sai_used)[:, None], pm)
    ss = np.maximum(np.sum((Pt - Pt.mean(1, keepdims=True))**2, 1), 1e-10)
    r2 = 1 - np.sum((Pp - Pt)**2, 1) / ss
    nr = 100 * np.sqrt(np.mean((Pp - Pt)**2, 1)) / np.maximum(Pt.max(1) - Pt.min(1), 1e-10)
    pk = np.abs(Pp.max(1) - Pt.max(1)) / Pt.max(1)
    return r2, nr, pk


def eos_accuracy(ph10):
    reg = ph10['regime'].values
    sai = ph10['SAI'].values
    r2_2, nr_2, pk_2 = _metrics(ph10, np.zeros(len(sai)))
    eps = np.random.default_rng(RNG_SEED).normal(0, SIGMA_SAI_EOS3, len(sai))
    r2_3, nr_3, pk_3 = _metrics(ph10, sai + eps)
    out = {}
    for name, m in [('R1', reg == 'R1'), ('R2', reg == 'R2'), ('R3', reg == 'R3'), ('All', reg != 'R3')]:
        out[name] = {'N': int(m.sum()),
                     'EOS2_R2': float(np.median(r2_2[m])), 'EOS2_NRMSE': float(np.median(nr_2[m])),
                     'EOS3_R2': float(np.median(r2_3[m])), 'EOS3_NRMSE': float(np.median(nr_3[m])),
                     'EOS2_peak10_pct': 100 * float(np.mean(pk_2[m] < 0.10))}
    # design law: NRMSE = k * sigma_SAI (origin-constrained, all R1-R3 curves as in Fig. 2).
    # k depends on the noise realisation (seed-to-seed SD ~1.3), so the median-NRMSE curve
    # is averaged over K_SEEDS realisations (default_rng(0..K_SEEDS-1)); k is linear in it.
    out['design_law'] = design_law(ph10, [np.random.default_rng(s) for s in range(K_SEEDS)])
    out['design_law_single'] = {
        'default_rng(42)': design_law(ph10, [np.random.default_rng(RNG_SEED)])['All']['k'],
        'np.random.seed(42)': design_law(ph10, ['legacy'])['All']['k']}
    return out


SIGMAS = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])


def design_law(ph10, gens):
    reg = ph10['regime'].values
    sai = ph10['SAI'].values
    med = {'All': np.zeros(len(SIGMAS)), 'R1': np.zeros(len(SIGMAS)), 'R2': np.zeros(len(SIGMAS))}
    k_each = []
    for g in gens:
        m_all = []
        for j, s in enumerate(SIGMAS):
            if s == 0:
                e = 0.0
            elif g == 'legacy':                       # previous Fig. 2 script: np.random.seed(42) per sigma
                np.random.seed(RNG_SEED)
                e = np.random.normal(0, s, len(sai))
            else:
                e = np.random.default_rng(g.bit_generator.seed_seq.entropy).normal(0, s, len(sai))
            _, nr, _ = _metrics(ph10, sai + e)
            med['All'][j] += np.median(nr) / len(gens)
            med['R1'][j] += np.median(nr[reg == 'R1']) / len(gens)
            med['R2'][j] += np.median(nr[reg == 'R2']) / len(gens)
            m_all.append(np.median(nr))
        x, y = SIGMAS[1:], np.array(m_all)[1:]
        k_each.append((x @ y) / (x @ x))
    out = {}
    for k, v in med.items():
        x, y = SIGMAS[1:], v[1:]
        kk = (x @ y) / (x @ x)
        out[k] = {'k': float(kk), 'R2': float(1 - np.sum((y - kk * x)**2) / np.sum((y - y.mean())**2)),
                  'median_nrmse': [float(t) for t in v]}
    out['All']['k_seed_sd'] = float(np.std(k_each, ddof=1)) if len(k_each) > 1 else 0.0
    out['n_seeds'] = len(gens)
    return out


# ══════════════════════════════════════════════════════════
# 5. I_opt VALIDATION
# ══════════════════════════════════════════════════════════
def iopt_validation(ph10):
    sensor = EOSSensor()
    table = {}
    for S_t in [5, 7, 10, 15, 20, 50]:
        v = sensor.validate_I_opt(alpha=0.05, Pmax=10.0, beta=0.05 / S_t)
        table[S_t] = {'I_num': v.I_opt_numeric, 'I_ana': v.I_opt_paper_form,
                      'dI_pct': v.delta_I_paper_form_pct, 'P_num': v.P_gross_at_numeric,
                      'dP_pct': v.delta_P_paper_form_pct,
                      'dP_pct_Aopt': v.delta_P_closed_form_pct}
    # dataset-wide (same grid rule as EOSSensor.validate_I_opt, vectorised)
    dP = []
    for _, c in ph10.iterrows():
        a, b, pm = c['alpha'], c['beta'], c['Pmax']
        Imax = max(5 * pm / b, 3 * pm / a, 5000.0)
        I = np.arange(1, 10001) * Imax / 10000
        Pn = ph10_gross(I, a, b, pm).max()
        Ia = (pm / a)**W_ALPHA * (pm / b)**W_BETA
        dP.append(100 * (Pn - ph10_gross(np.array([Ia]), a, b, pm)[0]) / Pn)
    dP = np.array(dP)
    reg = ph10['regime'].values
    ds = {r: {'N': int((reg == r).sum()), 'median_dP': float(np.median(dP[reg == r])),
              'max_dP': float(dP[reg == r].max()),
              'pct_dP<0.1': 100 * float(np.mean(dP[reg == r] < 0.1))} for r in ['R1', 'R2', 'R3']}
    return {'table': table, 'dataset': ds}


# ══════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════
def f(x, nd=3):
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(f(v, nd) for v in x) + ']'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f'{x:.2e}'
    return f'{x:.{nd}f}'


def judge(old, new, better):
    """better: 'low', 'high', 'none' (descriptive)."""
    if old is None or better == 'none' or isinstance(old, (list, tuple)):
        return ''
    if np.isclose(old, new, rtol=0.02, atol=1e-3):
        return '≈'
    return '改善' if (new < old) == (better == 'low') else '**悪化**'


def build_report(R):
    L = ['# 正本テーブル（seed=42, analysis/rerun_canonical.py）', '']
    d = R['data']
    L += ['## 1. N（フィルタ前後）', '',
          '| 段階 | 全体 | R1 | R2 | R3 |', '|---|---|---|---|---|',
          f"| Ph10 raw rows | {d['Ph10_raw']} | | | |",
          f"| Ph10 品質フィルタ後 (α,β>1e-6, Pmax>0.01) | {d['ph10']['N']} | {d['ph10']['R1']} | {d['ph10']['R2']} | {d['ph10']['R3']} |",
          f"| Ph11 raw → 品質フィルタ後 | {d['Ph11_raw']} → {d['ph11_N']} | | | |"]
    s = R['sai']['canonical']
    L += [f"| SAI 回復: 抽出 | {s['All']['n']} | {s['R1']['n']} | {s['R2']['n']} | 0 |",
          f"| SAI 回復: R²_fit ≥ {R2_MIN} 通過 | {s['All']['n_acc']} | {s['R1']['n_acc']} | {s['R2']['n_acc']} | — |",
          f"| Fitter innocence: 合成 → 収束 → R² 通過 | {R['fz']['fitter']['N_synth']} → {R['fz']['fitter']['converged']} → {R['fz']['fitter']['accepted']} | | | |",
          '']
    sd = R['sensitivity_dataset_r2']
    L += [f"感度分析（参考・正本外）: データセット側の piCurve R² ≥ {R2_MIN} を適用すると "
          f"N={sd['counts']['N']} (R1 {sd['counts']['R1']} / R2 {sd['counts']['R2']} / R3 {sd['counts']['R3']}), "
          f"slope={f(sd['scaling']['slope'])}, intercept={f(sd['scaling']['intercept'])}, r²={f(sd['scaling']['r2'])}, "
          f"FZ={sd['FZ_obs']}, perm(2000, β)={f(sd['perm']['mean'],1)}±{f(sd['perm']['sd'],1)}", '']

    sc = R['scaling']
    L += ['## 2. α–β スケーリング則 (OLS, N=%d)' % sc['N'], '',
          f"log10β = {sc['slope']:.4f}·log10α {sc['intercept']:+.4f}; r² = {sc['r2']:.4f}; "
          f"SE_slope = {sc['se_slope']:.4f}; SD(SAI) = {sc['sd_SAI']:.4f}", '']

    z = R['fz']
    ra = z['retention_adjusted']
    L += [f'## 3. 禁止域 (S_c = {FZ_S_C:g}, 半幅 {FZ_HALF_WIDTH_DEX} dex: {FZ_LO:.4f} ≤ S ≤ {FZ_HI:.4f}) と検定', '',
          '| 検定 | 結果 |', '|---|---|',
          f"| 観測件数 Ph10 | {z['poisson']['obs']} / {d['ph10']['N']} (S = {', '.join(f'{x:.3f}' for x in z['poisson']['obs_S'])}) |"]
    for k in ['perm10k_both', 'perm2k_beta']:
        p = z[k]
        L.append(f"| Permutation ({p['mode']}, B={p['B']}) | {f(p['mean'],1)} ± {f(p['sd'],1)}, median {f(p['median'],0)}, IQR {f(p['iqr'],0)}, Z={f(p['Z'],1)}, p(≤obs)={f(p['p_le_obs'],4)} |")
    fi = z['fitter']
    L += [f"| Bootstrap (B=10,000) | 99% CI {f(z['bootstrap']['ci99'],0)} |",
          f"| Poisson (全範囲一様) | λ={f(z['poisson']['full_range']['lambda'],1)}, p={f(z['poisson']['full_range']['p'])} |",
          f"| Fitter retention | FZ 内 {fi['pre_in_FZ']} → R² 通過 {fi['accepted']} 中 FZ 内 {fi['post_in_FZ']} (retention {f(fi['ratio'],3)}); 下へ {fi['post_below_FZ']}, 上へ {fi['post_above_FZ']} |",
          f"| 同・保持率補正後の帰無 | {f(ra['mean'],1)} ± {f(ra['sd'],1)}, P(≤obs)={f(ra['p_le_obs'],4)}; 観測を説明するのに要る保持率 ≤ {f(ra['retention_needed'],3)} |",
          f"| 同・旧手順（真値初期値 TRF） | FZ 内 {fi['legacy_post_in_FZ']} / {fi['N_synth']} (ratio {f(fi['legacy_ratio'],2)}) |",
          f"| 同・I_α ≥ 25 の曲線のみ | n={fi['identifiable_n']}, ratio {f(fi['identifiable_ratio'],2)} |",
          f"| Ph11（SI Test 5、表現特異性） | FZ 内 {z['ph11']} / {d['ph11_N']}; Ph11 自身の permutation 帰無 {f(z['ph11_perm']['mean'],1)} ± {f(z['ph11_perm']['sd'],1)}, Z={f(z['ph11_perm']['Z'],1)}, p(≤obs)={f(z['ph11_perm']['p_le_obs'],4)} |",
          f"| Gap 検出の bin 幅感度（記述のみ, S<3, n={z['binwidth']['n_S<3']}） | 下限 {f(z['binwidth']['lower_range'],2)}, 上限 {f(z['binwidth']['upper_range'],2)} |",
          '']

    L += ['## 4. SAI 回復（N=800, 3% ノイズ, 12 ステップ）', '',
          '| 方式 | 領域 | 通過/抽出 | MAD×1.4826 | SD | 外れ \\|ΔSAI\\|>0.5 | \\|ΔSAI\\|<0.15 (通過中) | 同 (抽出比) |',
          '|---|---|---|---|---|---|---|---|']
    names = {'legacy': '旧手順 (真値初期値・TRF・1start)', 'multistart_nofilter': 'multi-start LM (フィルタ無)',
             'canonical': '**正本: multi-start LM + R²≥0.95**'}
    for key, lab in names.items():
        for reg in ['R1', 'R2', 'All']:
            v = R['sai'][key][reg]
            L.append(f"| {lab} | {reg} | {v['n_acc']}/{v['n']} | {f(v['MAD'])} | {f(v['SD'])} | {f(v['outlier_pct'],1)}% | {f(v['cov015_pct'],1)}% | {f(v['cov015_of_sampled_pct'],1)}% |")
    L += ['', '誤差予算（正本）:', '', '| 領域 | MAD log α | MAD log β | SD log α | SD log β | 予測 σ_SAI (MAD 合成) | 実測 MAD |',
          '|---|---|---|---|---|---|---|']
    for reg in ['R1', 'R2', 'All']:
        v = s[reg]
        L.append(f"| {reg} | {f(v['MAD_la'])} | {f(v['MAD_lb'])} | {f(v['SD_la'])} | {f(v['SD_lb'])} | {f(v['pred_sigma_SAI_MAD'])} | {f(v['MAD'])} |")
    L.append('')

    e = R['eos']
    L += ['## 5. EOS 予測精度（P_gross, I=1–3000, σ_SAI=0.15）', '',
          '| 領域 | N | EOS2 R² | EOS2 NRMSE% | EOS3 R² | EOS3 NRMSE% | EOS2 ピーク±10% |', '|---|---|---|---|---|---|---|']
    for reg in ['R1', 'R2', 'R3', 'All']:
        v = e[reg]
        L.append(f"| {reg} | {v['N']} | {f(v['EOS2_R2'])} | {f(v['EOS2_NRMSE'],1)} | {f(v['EOS3_R2'])} | {f(v['EOS3_NRMSE'],1)} | {f(v['EOS2_peak10_pct'],1)}% |")
    dl = e['design_law']
    L += ['', f"設計則 NRMSE = k·σ_SAI（原点拘束, {dl['n_seeds']} ノイズ実現の平均）: " + '; '.join(
        f"{k}: k={dl[k]['k']:.1f}, R²={dl[k]['R2']:.4f}" for k in ['All', 'R1', 'R2'])
        + f"; seed 間 SD(k_All) = {dl['All']['k_seed_sd']:.2f}", '',
        '単一ノイズ実現での k（原因切り分け）: ' + '; '.join(f"{k}: {v:.2f}" for k, v in e['design_law_single'].items()), '']

    io = R['iopt']
    L += ['## 6. I_opt 検証（Pmax=10, α=0.05, 論文形）', '',
          '| S | I_num | I_ana | ΔI% | P_num | ΔP% | ΔP% (A_OPT 形) |', '|---|---|---|---|---|---|---|']
    for S_t, v in io['table'].items():
        L.append(f"| {S_t} | {v['I_num']:.0f} | {v['I_ana']:.0f} | {v['dI_pct']:.1f} | {v['P_num']:.3f} | {f(v['dP_pct'])} | {f(v['dP_pct_Aopt'])} |")
    L += ['', 'データセット全曲線（論文形）: ' + '; '.join(
        f"{r}: N={v['N']}, median ΔP={f(v['median_dP'])}%, max ΔP={f(v['max_dP'])}%, ΔP<0.1% は {f(v['pct_dP<0.1'],1)}%"
        for r, v in io['dataset'].items()), '']

    # ── diff ──
    L += ['## 7. 旧正本（第2ラウンド）との差分', '', '| 項目 | 旧 | 新 | 差 | 判定 |', '|---|---|---|---|---|']
    pm = R['fz']
    rows = [
        ('FZ 窓 [下限, 上限]', [OLD['FZ_lo'], OLD['FZ_hi']], [FZ_LO, FZ_HI], 'none'),
        ('N Ph10', OLD['N_ph10'], d['ph10']['N'], 'none'),
        ('scaling slope', OLD['slope'], sc['slope'], 'none'),
        ('FZ 観測 Ph10', OLD['FZ_obs'], pm['poisson']['obs'], 'none'),
        ('FZ Ph11', OLD['FZ_ph11'], pm['ph11'], 'none'),
        ('Poisson λ (全範囲)', OLD['poisson_full_lambda'], pm['poisson']['full_range']['lambda'], 'none'),
        ('Poisson p: 全範囲一様', OLD['poisson_full_p'], pm['poisson']['full_range']['p'], 'low'),
        ('Bootstrap 99% CI', OLD['boot_ci99'], pm['bootstrap']['ci99'], 'none'),
        ('Perm 10k both mean', OLD['perm10k_both_mean'], pm['perm10k_both']['mean'], 'none'),
        ('Perm 10k both sd', OLD['perm10k_both_sd'], pm['perm10k_both']['sd'], 'none'),
        ('Perm 10k both median', OLD['perm10k_both_median'], pm['perm10k_both']['median'], 'none'),
        ('Perm 10k both Z', OLD['perm10k_both_Z'], pm['perm10k_both']['Z'], 'low'),
        ('Perm 2k β mean ± sd', [OLD['perm2k_beta_mean'], OLD['perm2k_beta_sd']], [pm['perm2k_beta']['mean'], pm['perm2k_beta']['sd']], 'none'),
        ('Fitter retention', OLD['fitter_ratio'], pm['fitter']['ratio'], 'none'),
        ('Fitter accepted', OLD['fitter_accepted'], pm['fitter']['accepted'], 'none'),
        ('保持率補正後の帰無 mean', OLD['retadj_mean'], pm['retention_adjusted']['mean'], 'none'),
        ('保持率補正後の帰無 P(≤obs)', OLD['retadj_p'], pm['retention_adjusted']['p_le_obs'], 'low'),
    ]
    for reg in ['R1', 'R2']:
        v = s[reg]
        rows += [(f'SAI {reg} MAD', OLD[f'SAI_MAD_{reg}'], v['MAD'], 'low'),
                 (f'SAI {reg} SD', OLD[f'SAI_SD_{reg}'], v['SD'], 'low'),
                 (f'SAI {reg} |ΔSAI|<0.15 (%)', OLD[f'SAI_cov_{reg}'], v['cov015_pct'], 'high')]
    rows += [('誤差予算 MAD log α (All)', OLD['eb_MAD_la'], s['All']['MAD_la'], 'low'),
             ('誤差予算 MAD log β (All)', OLD['eb_MAD_lb'], s['All']['MAD_lb'], 'low')]
    for reg in ['R1', 'R2', 'All']:
        for m, b in [('EOS2_R2', 'high'), ('EOS2_NRMSE', 'low'), ('EOS3_R2', 'high'), ('EOS3_NRMSE', 'low')]:
            rows.append((f'{m} {reg}', OLD[f'{m}_{reg}'], e[reg][m], b))
    rows += [('設計則 k (All)', OLD['k_design'], e['design_law']['All']['k'], 'none'),
             ('設計則 R²', OLD['k_r2'], e['design_law']['All']['R2'], 'high'),
             ('EOS2 ピーク±10% R1', OLD['peak10_R1'], e['R1']['EOS2_peak10_pct'], 'high')]
    for name, old, new, b in rows:
        if isinstance(old, (list, tuple)):
            diff = ''
        else:
            diff = f(new - old)
        L.append(f"| {name} | {f(old)} | {f(new)} | {diff} | {judge(old, new, b)} |")
    L += ['', '判定は ±2% 以内を ≈ とした（旧 = 第2ラウンド正本）。']
    return '\n'.join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {}
    print('[1] data'); D = load_data(); ph10, ph11 = D['Ph10'], D['Ph11']
    R['data'] = {'Ph10_raw': D['Ph10_raw'], 'Ph11_raw': D['Ph11_raw'], 'ph10': counts(ph10),
                 'ph11_N': len(ph11)}
    print('[2] scaling law'); R['scaling'] = scaling_law(ph10)

    print('[3] forbidden zone tests')
    a, b = ph10['alpha'].values, ph10['beta'].values
    R['fz'] = {'poisson': poisson_tests(ph10), 'bootstrap': bootstrap_test(ph10),
               'perm10k_both': permutation_test(a, b, 10000, 'both', key='both'),
               'perm2k_beta': permutation_test(a, b, 2000, 'beta'),
               'fitter': fitter_innocence(ph10), 'ph11': fz_count(ph11['S'].values),
               'binwidth': binwidth_test(ph10)}
    # Ph11 against its OWN coupling-destroyed null (separate stream; main RNG untouched)
    g11 = np.random.default_rng(RNG_SEED)
    a11, b11 = ph11['alpha'].values, ph11['beta'].values
    c11 = np.array([fz_count(g11.permutation(a11) / g11.permutation(b11)) for _ in range(10000)])
    o11 = fz_count(a11 / b11)
    R['fz']['ph11_perm'] = {'obs': o11, 'mean': float(c11.mean()), 'sd': float(c11.std(ddof=1)),
                            'Z': float((o11 - c11.mean()) / c11.std(ddof=1)), 'p_le_obs': float(np.mean(c11 <= o11))}
    R['fz']['retention_adjusted'] = retention_adjusted_null(
        PERM_COUNTS['both'], R['fz']['poisson']['obs'], R['fz']['fitter']['ratio'])

    print('[4] SAI recovery'); res = sai_recovery(ph10)
    res.to_csv(os.path.join(OUT, 'sai_recovery.csv'), index=False)
    R['sai'] = {'legacy': sai_summary(res, 'legacy', False),
                'multistart_nofilter': sai_summary(res, 'multistart', False),
                'canonical': sai_summary(res, 'multistart', True)}

    print('[5] EOS accuracy + design law'); R['eos'] = eos_accuracy(ph10)
    print('[6] I_opt'); R['iopt'] = iopt_validation(ph10)

    print('[7] sensitivity: dataset-level R2 filter (reference only)')
    sub = ph10[pd.to_numeric(ph10['R2'], errors='coerce') >= R2_MIN]
    R['sensitivity_dataset_r2'] = {'counts': counts(sub), 'scaling': scaling_law(sub),
                                   'FZ_obs': fz_count(sub['S'].values),
                                   'perm': permutation_test(sub['alpha'].values, sub['beta'].values, 2000, 'beta')}

    with open(os.path.join(OUT, 'canonical.json'), 'w') as fh:
        json.dump(R, fh, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    rep = build_report(R)
    with open(os.path.join(OUT, 'canonical_tables.md'), 'w') as fh:
        fh.write(rep + '\n')
    print(rep)


if __name__ == '__main__':
    main()
