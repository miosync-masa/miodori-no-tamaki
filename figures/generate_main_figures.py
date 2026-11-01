import zipfile
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA SETUP
# ============================================================
with zipfile.ZipFile('raw_data/csv.zip') as z:
    df = pd.read_csv(z.open('Opt_ParVal_of_piModels.csv'))
ph10 = df[df['Model_piCurve_pkg'] == 'Ph10'].copy()
ph11 = df[df['Model_piCurve_pkg'] == 'Ph11'].copy()
for d in [ph10, ph11]:
    for c in ['alpha','beta','Pmax','R','R2adj']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
if 'shape' in ph11.columns:
    ph11['gamma_free'] = pd.to_numeric(ph11['shape'], errors='coerce')

valid = (ph10['alpha'] > 1e-6) & (ph10['beta'] > 1e-6) & (ph10['Pmax'] > 0.01)
ph10 = ph10[valid].copy()
ph10['R'] = ph10['R'].fillna(0)

alpha = ph10['alpha'].values
beta  = ph10['beta'].values
Pmax  = ph10['Pmax'].values
R_val = ph10['R'].values
N = len(ph10)
S = alpha / beta
GAMMA = np.cosh(1.0)**2

la = np.log10(alpha)
lb = np.log10(beta)
slope, intercept, _, _, _ = stats.linregress(la, lb)   # reported for verification
M_SLOPE, M_INTER = 0.814, -1.355                         # published scaling law (Main.tex Eq. 3)
beta_pred = 10**(M_SLOPE * la + M_INTER)
SAI = lb - (M_SLOPE * la + M_INTER)

# EOS3 benchmark noise: one realisation, default_rng(42), shared by Fig. 1 and Fig. 4
# (identical to analysis/rerun_canonical.py and figures/generate_si_figures.py)
SIGMA_SAI_EOS3 = 0.15
EPS_EOS3 = np.random.default_rng(42).normal(0, SIGMA_SAI_EOS3, N)

R1 = S > 10
R2 = (S > 3) & (S <= 10)
R3 = S <= 3

I_eval = np.linspace(1, 3000, 200)

def ph10_curve(I, Pm, a, b, R_):
    I_s = np.maximum(I, 1e-10)
    return Pm * np.tanh(a * I / max(Pm, 1e-10)) * \
           np.tanh((max(Pm, 1e-10) / (b * I_s))**GAMMA) - R_

def ph10_gross(I, Pm, a, b):
    I_s = np.maximum(I, 1e-10)
    return Pm * np.tanh(a * I / max(Pm, 1e-10)) * \
           np.tanh((max(Pm, 1e-10) / (b * I_s))**GAMMA)

def compute_metrics(eps):
    SAI_n = SAI + eps
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

# Compute metrics
nrmse_2var = np.zeros(N); r2_2var = np.zeros(N)
for i in range(N):
    P_true = ph10_curve(I_eval, Pmax[i], alpha[i], beta[i], R_val[i])
    P_2 = ph10_curve(I_eval, Pmax[i], alpha[i], beta_pred[i], R_val[i])
    P_range = max(P_true.max() - P_true.min(), 1e-10)
    ss_tot = np.sum((P_true - P_true.mean())**2)
    nrmse_2var[i] = np.sqrt(np.mean((P_2 - P_true)**2)) / P_range
    r2_2var[i] = 1 - np.sum((P_2 - P_true)**2) / max(ss_tot, 1e-10)

nrmse_3var, r2_3var = compute_metrics(EPS_EOS3)

# Gross versions for Fig 4
r2_eos2_gross = np.zeros(N)
r2_eos3_gross = np.zeros(N)
SAI_noisy = SAI + EPS_EOS3
for i in range(N):
    P_true = ph10_gross(I_eval, Pmax[i], alpha[i], beta[i])
    P_eos2 = ph10_gross(I_eval, Pmax[i], alpha[i], beta_pred[i])
    b3 = beta_pred[i] * 10**SAI_noisy[i]
    P_eos3 = ph10_gross(I_eval, Pmax[i], alpha[i], b3)
    ss_tot = np.sum((P_true - P_true.mean())**2)
    r2_eos2_gross[i] = 1 - np.sum((P_eos2 - P_true)**2) / max(ss_tot, 1e-10)
    r2_eos3_gross[i] = 1 - np.sum((P_eos3 - P_true)**2) / max(ss_tot, 1e-10)

# σ sweep for Fig 2: k depends on the noise realisation (seed-to-seed SD ≈ 1.3),
# so median NRMSE is averaged over K_SEEDS realisations (default_rng(0..K_SEEDS-1)),
# matching analysis/rerun_canonical.py.
K_SEEDS = 100
sigmas = np.array([0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
P_true_all = np.array([ph10_curve(I_eval, Pmax[i], alpha[i], beta[i], R_val[i]) for i in range(N)])
P_range_all = np.maximum(P_true_all.max(1) - P_true_all.min(1), 1e-10)

def sweep_nrmse(sig, seed):
    e = np.random.default_rng(seed).normal(0, sig, N) if sig > 0 else 0.0
    b_n = (beta_pred * 10**(SAI + e))[:, None]
    I_s = np.maximum(I_eval, 1e-10)
    P_pred = Pmax[:, None] * np.tanh(alpha[:, None] * I_eval / Pmax[:, None]) * \
             np.tanh((Pmax[:, None] / (b_n * I_s))**GAMMA) - R_val[:, None]
    return np.sqrt(np.mean((P_pred - P_true_all)**2, 1)) / P_range_all

median_nrmse_all = np.zeros(len(sigmas)); median_nrmse_R1 = np.zeros(len(sigmas)); median_nrmse_R2 = np.zeros(len(sigmas))
median_nrmse_seed = np.zeros((K_SEEDS, len(sigmas)))
for j, sig in enumerate(sigmas):
    for seed in range(K_SEEDS):
        nm = sweep_nrmse(sig, seed)
        median_nrmse_seed[seed, j] = np.median(nm)
        median_nrmse_all[j] += np.median(nm) / K_SEEDS
        median_nrmse_R1[j] += np.median(nm[R1]) / K_SEEDS
        median_nrmse_R2[j] += np.median(nm[R2]) / K_SEEDS

C_R1 = '#2196F3'; C_R2 = '#FF9800'; C_2var = '#E53935'; C_3var = '#1E88E5'

print(f"Data loaded: N={N}, R1={R1.sum()}, R2={R2.sum()}, R3={R3.sum()}")

# ============================================================
# FIG 1: EOS accuracy boxplot
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')

r2_2var_clip = np.clip(r2_2var, -0.5, 1.0)
r2_3var_clip = np.clip(r2_3var, -0.5, 1.0)

data_2var = [r2_2var_clip[R1], r2_2var_clip[R2], r2_2var_clip[R3]]
data_3var = [r2_3var_clip[R1], r2_3var_clip[R2], r2_3var_clip[R3]]

positions_2 = [0.8, 2.8, 4.8]; positions_3 = [1.2, 3.2, 5.2]

bp2 = ax1.boxplot(data_2var, positions=positions_2, widths=0.35,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=2))
bp3 = ax1.boxplot(data_3var, positions=positions_3, widths=0.35,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=2))
for patch in bp2['boxes']: patch.set_facecolor(C_2var); patch.set_alpha(0.7)
for patch in bp3['boxes']: patch.set_facecolor(C_3var); patch.set_alpha(0.7)

ax1.axhline(0.95, color='green', ls='--', lw=1, alpha=0.5)
ax1.axhline(0.90, color='green', ls=':', lw=1, alpha=0.3)

med_r1_2 = np.median(r2_2var[R1]); med_r1_3 = np.median(r2_3var[R1])
ax1.annotate(f'{med_r1_2:.3f}', xy=(0.8, med_r1_2), xytext=(0.15, med_r1_2-0.08),
             fontsize=9, color=C_2var, fontweight='bold')
ax1.annotate(f'{med_r1_3:.3f}', xy=(1.2, med_r1_3), xytext=(1.4, med_r1_3-0.08),
             fontsize=9, color=C_3var, fontweight='bold')

med_r2_2 = np.median(r2_2var[R2]); med_r2_3 = np.median(r2_3var[R2])
ax1.annotate('', xy=(3.2, med_r2_3), xytext=(2.8, med_r2_2),
             arrowprops=dict(arrowstyle='->', color='#E65100', lw=2.5))
ax1.text(3.55, (med_r2_2 + med_r2_3)/2, f'{med_r2_2:.3f} → {med_r2_3:.3f}\n+SAI rescues R2!',
         fontsize=10, color='#E65100', fontweight='bold', va='center')

ax1.set_xticks([1.0, 3.0, 5.0])
ax1.set_xticklabels([
    f'R1 — Factorized\n(S > 10, N={R1.sum()})',
    f'R2 — Transition\n(3 < S ≤ 10, N={R2.sum()})',
    f'R3 — Coupled\n(S ≤ 3, N={R3.sum()})'], fontsize=10)
ax1.set_ylabel('R²  (predicted vs actual PI curve)', fontsize=12)
ax1.set_ylim(-0.55, 1.05)
legend_elements = [
    Patch(facecolor=C_2var, alpha=0.7, label='2-var EOS: (α, Pmax) with β = β_pred(α)'),
    Patch(facecolor=C_3var, alpha=0.7, label='3-var EOS: (α, Pmax, SAI) assuming σ_SAI = 0.15')]
ax1.legend(handles=legend_elements, loc='lower left', fontsize=9.5, framealpha=0.9)
ax1.set_title('Fig. 1 — Equation of State: Prediction Accuracy by Regime',
              fontsize=13, fontweight='bold', pad=15)
fig1.tight_layout()
fig1.savefig('figures/Fig1_EOS_accuracy_by_regime.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Fig 1 saved")

# ============================================================
# FIG 2: Sensor design spec
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_facecolor('white')

ax2.plot(sigmas, median_nrmse_R1*100, 'o-', color=C_R1, lw=2, ms=8,
         label=f'R1 (S > 10, N={R1.sum()})', zorder=5)
ax2.plot(sigmas, median_nrmse_R2*100, 's-', color=C_R2, lw=2, ms=8,
         label=f'R2 (3 < S ≤ 10, N={R2.sum()})', zorder=5)
ax2.plot(sigmas, median_nrmse_all*100, '^-', color='#666666', lw=1.5, ms=6,
         label=f'All (N={N})', alpha=0.7, zorder=4)

# Origin-constrained regression
x = sigmas[1:]; y = median_nrmse_all[1:] * 100
k0 = (x @ y) / (x @ x)
k_R1 = (x @ (median_nrmse_R1[1:] * 100)) / (x @ x)
k_R2 = (x @ (median_nrmse_R2[1:] * 100)) / (x @ x)
k_seed_sd = np.std(median_nrmse_seed[:, 1:] * 100 @ x / (x @ x), ddof=1)
x_line = np.linspace(0, 0.35, 100)
ax2.plot(x_line, k0 * x_line, 'k:', lw=2.0, alpha=0.7,
         label=f'NRMSE = {k0:.1f} · σ_SAI  (origin-constrained)')

y_pred = k0 * x; r2_fit = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

sigma_5 = 5.0 / k0
rect1 = Rectangle((0, 0), sigma_5, 5, linewidth=2, linestyle='--',
                    facecolor='#C8E6C9', edgecolor='#2E7D32', alpha=0.35, zorder=1)
ax2.add_patch(rect1)
ax2.annotate(f'Design target:\nNRMSE < 5%\nσ_SAI < {sigma_5:.3f}',
             xy=(sigma_5/2, 7.5), fontsize=9, color='#2E7D32', fontweight='bold', ha='center', zorder=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2E7D32', alpha=0.9))

sigma_10 = 10.0 / k0
rect2 = Rectangle((0, 0), sigma_10, 10, linewidth=2, linestyle='--',
                    facecolor='#FFF9C4', edgecolor='#F57F17', alpha=0.25, zorder=0)
ax2.add_patch(rect2)
ax2.annotate(f'NRMSE < 10%\nσ_SAI < {sigma_10:.3f}',
             xy=(0.255, 6.0), fontsize=9, color='#F57F17', fontweight='bold', ha='center', zorder=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#F57F17', alpha=0.9))

ax2.set_xlabel('σ_SAI  (measurement uncertainty of Stress Adaptation Index)', fontsize=11)
ax2.set_ylabel('Median NRMSE  (%)', fontsize=11)
ax2.set_xlim(-0.01, 0.33); ax2.set_ylim(-0.5, 18)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.set_title('Fig. 2 — Sensor Design Specification: NRMSE = k · σ_SAI  (approximately regime-invariant)',
              fontsize=13, fontweight='bold', pad=15)
ax2.text(0.97, 0.03,
         f'Origin-constrained fit:\nk = {k0:.1f}  (R² = {r2_fit:.4f}; mean of {K_SEEDS} noise realisations,\n'
         f'seed SD {k_seed_sd:.1f} = {100 * k_seed_sd / k0:.1f}%)\n'
         f'Approximately regime-invariant: k_R1 = {k_R1:.1f}, k_R2 = {k_R2:.1f} (ratio {k_R1 / k_R2:.2f})\n'
         f'γ₀ = cosh²(1) fixed',
         transform=ax2.transAxes, fontsize=10, va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.9))
fig2.tight_layout()
fig2.savefig('figures/Fig2_sensor_design_spec.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Fig 2 saved (k={k0:.1f}, R²={r2_fit:.6f})")

# ============================================================
# FIG 4: Representative curve overlays — FIXED line styles
# ============================================================
fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
fig4.patch.set_facecolor('white')

# Pick representative curves by a noise-independent rule: EOS2 R² uses beta_pred only,
# so it does not depend on the SAI noise draw. Each panel shows the curve in its regime
# whose EOS2 R² is closest to the stated percentile of that regime's EOS2 R² distribution.
def pick_by_eos2_percentile(mask, q):
    ids = np.where(mask)[0]
    v = r2_eos2_gross[ids]
    return ids[np.argmin(np.abs(v - np.percentile(v, q)))]

Q_A, Q_B, Q_C, Q_D = 90, 50, 50, 10
r1_best_pick = pick_by_eos2_percentile(R1, Q_A)
r1_pick = pick_by_eos2_percentile(R1, Q_B)
r2_pick = pick_by_eos2_percentile(R2, Q_C)
r2_drama_pick = pick_by_eos2_percentile(R2, Q_D)

picks = [
    (r1_best_pick, f'R1 — Factorized (EOS2 R² at R1 {Q_A}th pct)', '(a)'),
    (r1_pick, f'R1 — Factorized (EOS2 R² at R1 {Q_B}th pct)', '(b)'),
    (r2_pick, f'R2 — Transition (EOS2 R² at R2 {Q_C}th pct)', '(c)'),
    (r2_drama_pick, f'R2 — Transition (EOS2 R² at R2 {Q_D}th pct)', '(d)'),
]

I_plot = np.linspace(1, 2500, 300)

for ax, (idx, title, panel) in zip(axes.flat, picks):
    a, b, Pm = alpha[idx], beta[idx], Pmax[idx]
    s_val = S[idx]; bp = beta_pred[idx]; sai_val = SAI[idx]; sai_meas = SAI_noisy[idx]
    
    P_ref = ph10_gross(I_plot, Pm, a, b)
    P_eos2 = ph10_gross(I_plot, Pm, a, bp)
    b3 = bp * 10**sai_meas          # SAI measured with sigma_SAI = 0.15 (same draw as the legend R²)
    P_eos3 = ph10_gross(I_plot, Pm, a, b3)
    
    # ★ FIX: Reference を点線(dotted)に → 青EOS3と被っても見分けがつく
    ax.plot(I_plot, P_ref, 'k:', lw=3.0, label='Reference (Ph10 fit)', zorder=5)
    ax.plot(I_plot, P_eos2, '--', color='#E53935', lw=2.0, 
            label=f'EOS2 (R²={r2_eos2_gross[idx]:.3f})', zorder=4)
    ax.plot(I_plot, P_eos3, '-', color='#1E88E5', lw=2.0, alpha=0.85,
            label=f'EOS3 (R²={r2_eos3_gross[idx]:.3f})', zorder=3)
    
    # closed-form I_opt (Main Eq. 12, A_OPT = 1): I_alpha^(1/(γ0+1)) · I_beta^(γ0/(γ0+1))
    I_a = Pm / a
    iopt_ref = I_a**(1 / (GAMMA + 1)) * (Pm / b)**(GAMMA / (GAMMA + 1))
    iopt_eos2 = I_a**(1 / (GAMMA + 1)) * (Pm / bp)**(GAMMA / (GAMMA + 1))
    ax.axvline(iopt_ref, color='#424242', ls='-.', lw=1.5,
               label=f'I_opt reference ({iopt_ref:.0f})', zorder=2)
    ax.axvline(iopt_eos2, color='#E53935', ls='-.', lw=1.5,
               label=f'I_opt EOS2 ({iopt_eos2:.0f})', zorder=2)

    ax.set_xlabel('I (µmol m⁻² s⁻¹)', fontsize=10)
    ax.set_ylabel('P_gross', fontsize=10)
    ax.set_title(f'{panel} {title}\nS = {s_val:.1f}, SAI = {sai_val:+.2f}', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.set_xlim(0, 2500)
    ax.set_ylim(bottom=0)

fig4.suptitle('Fig. 4 — Representative PI curves: Reference vs EOS2 vs EOS3',
              fontsize=14, fontweight='bold', y=1.01)
fig4.tight_layout()
fig4.savefig('figures/Fig4_curve_overlays.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Fig 4 saved (Reference=dotted black, EOS2=dashed red, EOS3=solid blue)")

# ============================================================
# VERIFICATION
# ============================================================
print(f"\n{'='*60}")
print("VERIFICATION")
print(f"{'='*60}")
print(f"  Slope = {slope:.3f}, Intercept = {intercept:.3f}")
print(f"  k (origin-constrained) = {k0:.2f}; k_R1 = {k_R1:.2f}, k_R2 = {k_R2:.2f} (ratio {k_R1 / k_R2:.3f}); seed SD = {k_seed_sd:.2f} ({100 * k_seed_sd / k0:.1f}%)")
print(f"  R² (fit) = {r2_fit:.6f}")
print(f"  σ_SAI < {sigma_5:.3f} for NRMSE < 5%")
print(f"  σ_SAI < {sigma_10:.3f} for NRMSE < 10%")
print(f"  EOS2 R² median: R1={np.median(r2_2var[R1]):.3f}, R2={np.median(r2_2var[R2]):.3f}")
print(f"  EOS3 R² median: R1={np.median(r2_3var[R1]):.3f}, R2={np.median(r2_3var[R2]):.3f}")
for name, idx in zip(['(a)', '(b)', '(c)', '(d)'], [p[0] for p in picks]):
    print(f"  Fig4 {name}: {ph10['pi_number'].values[idx]}  S={S[idx]:.1f}  SAI={SAI[idx]:+.2f}  "
          f"EOS2 R²={r2_eos2_gross[idx]:.3f}  EOS3 R²={r2_eos3_gross[idx]:.3f}  "
          f"I_alpha={Pmax[idx] / alpha[idx]:.0f}  I_beta={Pmax[idx] / beta[idx]:.0f}  "
          f"I_beta_pred={Pmax[idx] / beta_pred[idx]:.0f}")
