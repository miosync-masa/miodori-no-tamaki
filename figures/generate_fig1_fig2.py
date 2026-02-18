import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA SETUP
# ============================================================
df = pd.read_csv('raw_data/Opt_ParVal_of_piModels.csv')
ph10 = df[df['Model_piCurve_pkg'] == 'Ph10'].copy()
for c in ['alpha','beta','Pmax','R']:
    ph10[c] = pd.to_numeric(ph10[c], errors='coerce')
valid = (ph10['alpha'] > 1e-6) & (ph10['beta'] > 1e-6) & (ph10['Pmax'] > 0.01)
ph10 = ph10[valid].copy()
ph10['R'] = ph10['R'].fillna(0)

N = len(ph10)
alpha = ph10['alpha'].values
beta  = ph10['beta'].values
Pmax  = ph10['Pmax'].values
R_val = ph10['R'].values
S = alpha / beta
GAMMA = np.cosh(1.0)**2

la = np.log10(alpha)
lb = np.log10(beta)
slope, intercept, _, _, _ = stats.linregress(la, lb)
beta_pred = 10**(slope * la + intercept)
SAI = lb - (slope * la + intercept)

I_eval = np.linspace(1, 3000, 200)

def ph10_curve(I, Pm, a, b, R_):
    I_s = np.maximum(I, 1e-10)
    return Pm * np.tanh(a * I / max(Pm, 1e-10)) * \
           np.tanh((max(Pm, 1e-10) / (b * I_s))**GAMMA) - R_

def compute_metrics(sigma_sai, seed=42):
    np.random.seed(seed)
    SAI_n = SAI + np.random.normal(0, sigma_sai, N) if sigma_sai > 0 else SAI.copy()
    nrmse = np.zeros(N)
    r2 = np.zeros(N)
    for i in range(N):
        P_true = ph10_curve(I_eval, Pmax[i], alpha[i], beta[i], R_val[i])
        beta_n = beta_pred[i] * 10**SAI_n[i]
        P_pred = ph10_curve(I_eval, Pmax[i], alpha[i], beta_n, R_val[i])
        P_range = max(P_true.max() - P_true.min(), 1e-10)
        ss_tot = np.sum((P_true - P_true.mean())**2)
        nrmse[i] = np.sqrt(np.mean((P_pred - P_true)**2)) / P_range
        r2[i] = 1 - np.sum((P_pred - P_true)**2) / max(ss_tot, 1e-10)
    return nrmse, r2

# 2-var (β from scaling only)
nrmse_2var = np.zeros(N)
r2_2var = np.zeros(N)
for i in range(N):
    P_true = ph10_curve(I_eval, Pmax[i], alpha[i], beta[i], R_val[i])
    P_2 = ph10_curve(I_eval, Pmax[i], alpha[i], beta_pred[i], R_val[i])
    P_range = max(P_true.max() - P_true.min(), 1e-10)
    ss_tot = np.sum((P_true - P_true.mean())**2)
    nrmse_2var[i] = np.sqrt(np.mean((P_2 - P_true)**2)) / P_range
    r2_2var[i] = 1 - np.sum((P_2 - P_true)**2) / max(ss_tot, 1e-10)

# 3-var with σ_SAI = 0.15
nrmse_3var, r2_3var = compute_metrics(0.15)

R1_mask = S > 10
R2_mask = (S > 3) & (S <= 10)
R3_mask = S <= 3

# Multiple σ levels for Fig 2
sigmas = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
median_nrmse_all = []
median_nrmse_R1 = []
median_nrmse_R2 = []
for sig in sigmas:
    nm, _ = compute_metrics(sig, seed=42)
    median_nrmse_all.append(np.median(nm))
    median_nrmse_R1.append(np.median(nm[R1_mask]))
    median_nrmse_R2.append(np.median(nm[R2_mask]))

median_nrmse_all = np.array(median_nrmse_all)
median_nrmse_R1 = np.array(median_nrmse_R1)
median_nrmse_R2 = np.array(median_nrmse_R2)

C_R1 = '#2196F3'
C_R2 = '#FF9800'
C_2var = '#E53935'
C_3var = '#1E88E5'

# ============================================================
# FIG 1 (FIXED): σ_SAI=0.15 explicitly in legend
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')

r2_2var_clip = np.clip(r2_2var, -0.5, 1.0)
r2_3var_clip = np.clip(r2_3var, -0.5, 1.0)

data_2var = [r2_2var_clip[R1_mask], r2_2var_clip[R2_mask], r2_2var_clip[R3_mask]]
data_3var = [r2_3var_clip[R1_mask], r2_3var_clip[R2_mask], r2_3var_clip[R3_mask]]

positions_2 = [0.8, 2.8, 4.8]
positions_3 = [1.2, 3.2, 5.2]

bp2 = ax1.boxplot(data_2var, positions=positions_2, widths=0.35,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=2))
bp3 = ax1.boxplot(data_3var, positions=positions_3, widths=0.35,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=2))

for patch in bp2['boxes']:
    patch.set_facecolor(C_2var); patch.set_alpha(0.7)
for patch in bp3['boxes']:
    patch.set_facecolor(C_3var); patch.set_alpha(0.7)

ax1.axhline(0.95, color='green', ls='--', lw=1, alpha=0.5)
ax1.axhline(0.90, color='green', ls=':', lw=1, alpha=0.3)

# R1 annotations
med_r1_2 = np.median(r2_2var[R1_mask])
med_r1_3 = np.median(r2_3var[R1_mask])
ax1.annotate(f'{med_r1_2:.3f}', xy=(0.8, med_r1_2), xytext=(0.15, med_r1_2-0.08),
             fontsize=9, color=C_2var, fontweight='bold')
ax1.annotate(f'{med_r1_3:.3f}', xy=(1.2, med_r1_3), xytext=(1.4, med_r1_3-0.08),
             fontsize=9, color=C_3var, fontweight='bold')

# R2 — dramatic jump with arrow
med_r2_2 = np.median(r2_2var[R2_mask])
med_r2_3 = np.median(r2_3var[R2_mask])
ax1.annotate('', xy=(3.2, med_r2_3), xytext=(2.8, med_r2_2),
             arrowprops=dict(arrowstyle='->', color='#E65100', lw=2.5))
ax1.text(3.55, (med_r2_2 + med_r2_3)/2, f'{med_r2_2:.2f} → {med_r2_3:.2f}\n+SAI rescues R2!',
         fontsize=10, color='#E65100', fontweight='bold', va='center')

ax1.set_xticks([1.0, 3.0, 5.0])
ax1.set_xticklabels([
    f'R1 — Factorized\n(S > 10, N={R1_mask.sum()})',
    f'R2 — Transition\n(3 < S ≤ 10, N={R2_mask.sum()})',
    f'R3 — Coupled\n(S ≤ 3, N={R3_mask.sum()})',
], fontsize=10)
ax1.set_ylabel('R²  (predicted vs actual PI curve)', fontsize=12)
ax1.set_ylim(-0.55, 1.05)

# ★ FIX: Legend with explicit σ_SAI assumption
legend_elements = [
    Patch(facecolor=C_2var, alpha=0.7, 
          label='2-var EOS: (α, Pmax) with β = β_pred(α)'),
    Patch(facecolor=C_3var, alpha=0.7, 
          label='3-var EOS: (α, Pmax, SAI) assuming σ_SAI = 0.15'),
]
ax1.legend(handles=legend_elements, loc='lower left', fontsize=9.5,
           framealpha=0.9)

ax1.set_title('Fig. 1 — Equation of State: Prediction Accuracy by Regime',
              fontsize=13, fontweight='bold', pad=15)

fig1.tight_layout()
fig1.savefig('raw_data/fig1_eos_boxplot_v2.png', dpi=250, bbox_inches='tight',
             facecolor='white')
print("Fig 1 v2 saved!")

# ============================================================
# FIG 2 (FIXED): origin-constrained regression + correct Design Box
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_facecolor('white')

ax2.plot(sigmas, median_nrmse_R1*100, 'o-', color=C_R1, lw=2, ms=8,
         label=f'R1 (S > 10, N={R1_mask.sum()})', zorder=5)
ax2.plot(sigmas, median_nrmse_R2*100, 's-', color=C_R2, lw=2, ms=8,
         label=f'R2 (3 < S ≤ 10, N={R2_mask.sum()})', zorder=5)
ax2.plot(sigmas, median_nrmse_all*100, '^-', color='#666666', lw=1.5, ms=6,
         label=f'All (N={N})', alpha=0.7, zorder=4)

# ★ FIX: Origin-constrained regression (NRMSE = k₀ · σ_SAI)
x = sigmas[1:]  # exclude σ=0
y = median_nrmse_all[1:] * 100
k0 = (x @ y) / (x @ x)  # forced through origin

x_line = np.linspace(0, 0.35, 100)
ax2.plot(x_line, k0 * x_line, 'k--', lw=1.5, alpha=0.6,
         label=f'NRMSE = {k0:.1f} · σ_SAI  (origin-constrained)')

# Verify fit quality
y_pred = k0 * x
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_fit = 1 - ss_res / ss_tot

# ★ FIX: Design Boxes — correct width (NO /100!)
# Box 1: NRMSE < 5% → σ_SAI < 5/k0
sigma_5 = 5.0 / k0
rect1 = Rectangle((0, 0), sigma_5, 5, linewidth=2, linestyle='--',
                    facecolor='#C8E6C9', edgecolor='#2E7D32',
                    alpha=0.35, zorder=1)
ax2.add_patch(rect1)
ax2.annotate(f'Design target:\nNRMSE < 5%\nσ_SAI < {sigma_5:.2f}',
             xy=(sigma_5/2, 2.5), fontsize=9, color='#2E7D32',
             fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#2E7D32', alpha=0.9))

# Box 2: NRMSE < 10% → σ_SAI < 10/k0
sigma_10 = 10.0 / k0
rect2 = Rectangle((0, 0), sigma_10, 10, linewidth=2, linestyle='--',
                    facecolor='#FFF9C4', edgecolor='#F57F17',
                    alpha=0.25, zorder=0)
ax2.add_patch(rect2)
ax2.annotate(f'NRMSE < 10%\nσ_SAI < {sigma_10:.2f}',
             xy=(sigma_10*0.7, 8.0), fontsize=9, color='#F57F17',
             fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#F57F17', alpha=0.9))

ax2.set_xlabel('σ_SAI  (measurement uncertainty of Stress Adaptation Index)', fontsize=11)
ax2.set_ylabel('Median NRMSE  (%)', fontsize=11)
ax2.set_xlim(-0.01, 0.33)
ax2.set_ylim(-0.5, 18)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)

ax2.set_title('Fig. 2 — Sensor Design Specification: NRMSE = k · σ_SAI  (regime-invariant)',
              fontsize=13, fontweight='bold', pad=15)

# Key message box
ax2.text(0.97, 0.03,
         f'Origin-constrained fit:\n'
         f'k = {k0:.1f}  (R² = {r2_fit:.6f})\n'
         f'Regime-invariant (R1 ≈ R2)\n'
         f'γ₀ = cosh²(1) fixed',
         transform=ax2.transAxes, fontsize=10, va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                   edgecolor='#1565C0', alpha=0.9))

fig2.tight_layout()
fig2.savefig('raw_data/fig2_design_spec_v2.png', dpi=250, bbox_inches='tight',
             facecolor='white')
print(f"Fig 2 v2 saved!")

# ============================================================
# VERIFY
# ============================================================
print(f"\n{'='*70}")
print("VERIFICATION")
print(f"{'='*70}")
print(f"  Origin-constrained k = {k0:.2f}")
print(f"  R² (origin-constrained) = {r2_fit:.6f}")
print(f"  Design Box 1: NRMSE < 5%  → σ_SAI < {sigma_5:.3f}")
print(f"  Design Box 2: NRMSE < 10% → σ_SAI < {sigma_10:.3f}")
print(f"\n  Cross-check: k × 0.10 = {k0 * 0.10:.2f}%  (should be ~5%)")
print(f"  Cross-check: k × 0.20 = {k0 * 0.20:.2f}%  (should be ~10%)")

# Per-regime k
x_r1 = sigmas[1:]
y_r1 = median_nrmse_R1[1:] * 100
k_r1 = (x_r1 @ y_r1) / (x_r1 @ x_r1)

y_r2 = median_nrmse_R2[1:] * 100
k_r2 = (x_r1 @ y_r2) / (x_r1 @ x_r1)

print(f"\n  Per-regime k (origin-constrained):")
print(f"    k_R1 = {k_r1:.2f}")
print(f"    k_R2 = {k_r2:.2f}")
print(f"    k_all = {k0:.2f}")
print(f"    k_R1/k_R2 = {k_r1/k_r2:.3f}  (≈1 = regime-invariant)")
