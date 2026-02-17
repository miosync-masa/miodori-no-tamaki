import pandas as pd
import numpy as np
from scipy import stats

ph10 = pd.read_csv('ph10_decomposed.csv', index_col=0)

# =====================================================
# R3 as PHASE TRANSITION: plateau_width = order parameter
# =====================================================
print("=== PHASE TRANSITION ANALYSIS ===")
print("Order parameter: plateau_width = log10(Iβ/Iα)")
print("                 equivalently SAI (r = -0.978)")

# Is there a sharp transition in R²adj as function of separation_ratio?
sep = ph10['separation_ratio'].values
r2 = ph10['R2adj'].values

# Bin by separation ratio
bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 200.0]
ph10['sep_bin'] = pd.cut(ph10['separation_ratio'], bins=bins)

print(f"\n--- R²adj vs Separation Ratio (Iβ/Iα) ---")
print(f"{'Iβ/Iα range':>20} {'N':>5} {'R²adj median':>13} {'R²adj mean':>11} {'R²adj σ':>9}")
for cat in ph10['sep_bin'].cat.categories:
    sub = ph10[ph10['sep_bin'] == cat]
    if len(sub) == 0: continue
    print(f"  {str(cat):>18} {len(sub):5d} {sub['R2adj'].median():13.4f} {sub['R2adj'].mean():11.4f} {sub['R2adj'].std():9.4f}")

# Sharp transition around Iβ/Iα ≈ 3?
below3 = ph10[ph10['separation_ratio'] < 3]
above3 = ph10[ph10['separation_ratio'] >= 3]
mw_stat, mw_p = stats.mannwhitneyu(below3['R2adj'], above3['R2adj'])
print(f"\nMann-Whitney (below vs above Iβ/Iα=3): U={mw_stat:.0f}, p={mw_p:.2e}")
print(f"  Below 3: median R²adj = {below3['R2adj'].median():.4f} (N={len(below3)})")
print(f"  Above 3: median R²adj = {above3['R2adj'].median():.4f} (N={len(above3)})")

# Model spread also transitions?
print(f"\n--- Model spread (σ of R²adj across 16 models) ---")
print(f"  Below Iβ/Iα=3: median σ = {below3['r2_model_spread'].median():.4f}")
print(f"  Above Iβ/Iα=3: median σ = {above3['r2_model_spread'].median():.4f}")

# =====================================================
# Phase diagram: 2D (Iα, Iβ) with phase boundary
# =====================================================
print(f"\n=== PHASE DIAGRAM ===")
print(f"In (log Iα, log Iβ) space:")
print(f"  Factorized phase: Iβ > Iα (above diagonal)")
print(f"  Coupled phase:    Iβ < Iα (below diagonal)")
print(f"  Phase boundary:   Iβ = Iα (diagonal)")
print(f"  Actual boundary:  Iβ/Iα ≈ 3 (shifted from naive)")

# What fraction lies in each quadrant?
log_Ia = np.log10(ph10['I_alpha'])
log_Ib = np.log10(ph10['I_beta_calc'])

print(f"\n  Iβ > 10×Iα (deep factorized):  {(ph10['separation_ratio'] > 10).mean():.1%}")
print(f"  3 < Iβ/Iα < 10 (marginal):     {((ph10['separation_ratio'] >= 3) & (ph10['separation_ratio'] < 10)).mean():.1%}")
print(f"  1 < Iβ/Iα < 3 (near-critical):  {((ph10['separation_ratio'] >= 1) & (ph10['separation_ratio'] < 3)).mean():.1%}")
print(f"  Iβ/Iα < 1 (coupled):            {(ph10['separation_ratio'] < 1).mean():.1%}")

# R3 organisms: are they a distinct cluster or continuous tail?
print(f"\n=== CONTINUOUS vs DISCRETE TRANSITION ===")
# Check: is there a GAP in separation_ratio distribution?
sep_sorted = np.sort(ph10['separation_ratio'].values)
gaps = np.diff(sep_sorted[:50])  # look at the low end
print(f"Smallest 20 separation ratios:")
for i in range(20):
    print(f"  {sep_sorted[i]:.3f}", end="")
print()

# Kolmogorov-Smirnov: test if R3 is from a different distribution
from scipy.stats import ks_2samp
r3_r2 = ph10[ph10['regime'] == 'R3_coupled']['R2adj'].values
r2_non_r3 = ph10[ph10['regime'] == 'R2_adaptive']['R2adj'].values  # compare to R2 (closest)
ks, ks_p = ks_2samp(r3_r2, r2_non_r3)
print(f"\nKS test (R3 vs R2 R²adj distributions): D={ks:.3f}, p={ks_p:.3e}")

# =====================================================
# Critical exponent: how does R²adj scale near boundary?
# =====================================================
print(f"\n=== SCALING NEAR PHASE BOUNDARY ===")
# R²adj ∝ (Iβ/Iα - 1)^ν near boundary?
near_boundary = ph10[(ph10['separation_ratio'] > 0.3) & (ph10['separation_ratio'] < 30)].copy()
near_boundary['log_sep'] = np.log10(near_boundary['separation_ratio'])
near_boundary['log_deficit'] = np.log10(1 - near_boundary['R2adj'] + 1e-6)  # log(1-R²)

# Linear regression in log-log space
valid = np.isfinite(near_boundary['log_sep']) & np.isfinite(near_boundary['log_deficit'])
r_crit, p_crit = stats.pearsonr(near_boundary.loc[valid, 'log_sep'], 
                                  near_boundary.loc[valid, 'log_deficit'])
slope, intercept, _, _, _ = stats.linregress(near_boundary.loc[valid, 'log_sep'], 
                                               near_boundary.loc[valid, 'log_deficit'])
print(f"log(1-R²) vs log(Iβ/Iα):")
print(f"  slope (critical exponent ν) = {slope:.3f}")
print(f"  r = {r_crit:.3f}, p = {p_crit:.2e}")
print(f"  → (1-R²) ~ (Iβ/Iα)^{slope:.2f}")
