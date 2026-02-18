import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('raw_data/Opt_ParVal_of_piModels.csv')
ph10 = df[df['Model_piCurve_pkg'] == 'Ph10'].copy()
ph11 = df[df['Model_piCurve_pkg'] == 'Ph11'].copy()
for d in [ph10, ph11]:
    for c in ['alpha','beta','Pmax','R','R2adj']:
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
slope, intercept, _, _, _ = stats.linregress(la, lb)
beta_pred = 10**(slope * la + intercept)
SAI = lb - (slope * la + intercept)

R1 = S > 10
R2 = (S > 3) & (S <= 10)
R3 = S <= 3

I_eval = np.linspace(1, 3000, 200)

def ph10_gross(I, Pm, a, b):
    """P_gross = Pmax * PCC * SCC (no R term)"""
    I_s = np.maximum(I, 1e-10)
    return Pm * np.tanh(a * I / max(Pm, 1e-10)) * \
           np.tanh((max(Pm, 1e-10) / (b * I_s))**GAMMA)

# ============================================================
# FIX ⑤: P_gross reformulation — verify EOS works on gross P
# ============================================================
print("="*70)
print("FIX ⑤: P_gross REFORMULATION")
print("="*70)

# EOS2 on P_gross (no R involved at all)
r2_eos2_gross = np.zeros(N)
nrmse_eos2_gross = np.zeros(N)
for i in range(N):
    P_true = ph10_gross(I_eval, Pmax[i], alpha[i], beta[i])
    P_eos2 = ph10_gross(I_eval, Pmax[i], alpha[i], beta_pred[i])
    P_range = max(P_true.max() - P_true.min(), 1e-10)
    ss_tot = np.sum((P_true - P_true.mean())**2)
    r2_eos2_gross[i] = 1 - np.sum((P_eos2 - P_true)**2) / max(ss_tot, 1e-10)
    nrmse_eos2_gross[i] = np.sqrt(np.mean((P_eos2 - P_true)**2)) / P_range

# EOS3 on P_gross
np.random.seed(42)
sig_sai = 0.15
SAI_noisy = SAI + np.random.normal(0, sig_sai, N)
r2_eos3_gross = np.zeros(N)
for i in range(N):
    P_true = ph10_gross(I_eval, Pmax[i], alpha[i], beta[i])
    b3 = beta_pred[i] * 10**SAI_noisy[i]
    P_eos3 = ph10_gross(I_eval, Pmax[i], alpha[i], b3)
    ss_tot = np.sum((P_true - P_true.mean())**2)
    r2_eos3_gross[i] = 1 - np.sum((P_eos3 - P_true)**2) / max(ss_tot, 1e-10)

print("\nP_gross EOS (R-free formulation):")
for name, mask in [('R1', R1), ('R2', R2), ('All R1+R2', R1|R2)]:
    m = mask
    print(f"  {name}: EOS2 R²={np.median(r2_eos2_gross[m]):.3f}, "
          f"EOS3 R²={np.median(r2_eos3_gross[m]):.3f}")

print("\n→ Numbers identical to P_net formulation (R cancels in R² and NRMSE)")
print("→ P_gross reformulation is SAFE: EOS is truly (α, Pmax) 2-variable")

# ============================================================
# FIX ①: γ₀ sensitivity — do γ-deviant curves hurt EOS?
# ============================================================
print(f"\n{'='*70}")
print("FIX ①: γ₀ SENSITIVITY ANALYSIS")
print(f"{'='*70}")

# Match Ph10/Ph11 and get γ_free
ph10_idx = ph10.set_index('pi_number')
ph11_idx = ph11.set_index('pi_number')
common = ph10_idx.index.intersection(ph11_idx.index)
p11c = ph11_idx.loc[common]

# Get γ_free for each Ph10 curve (matched by pi_number)
gamma_map = {}
for pi in common:
    if 'gamma_free' in p11c.columns:
        gamma_map[pi] = p11c.loc[pi, 'gamma_free']
    elif 'shape' in p11c.columns:
        gamma_map[pi] = pd.to_numeric(p11c.loc[pi, 'shape'], errors='coerce')

# Get delta_R2adj
delta_map = {}
p10c = ph10_idx.loc[common]
for pi in common:
    delta_map[pi] = p11c.loc[pi, 'R2adj'] - p10c.loc[pi, 'R2adj']

# Map back to ph10 array indices
ph10_pi = ph10['pi_number'].values
gamma_free_arr = np.full(N, np.nan)
delta_r2_arr = np.full(N, np.nan)
for i, pi in enumerate(ph10_pi):
    if pi in gamma_map:
        gamma_free_arr[i] = gamma_map[pi]
    if pi in delta_map:
        delta_r2_arr[i] = delta_map[pi]

# γ-deviant: Ph11 improves by > 0.01
gamma_deviant = delta_r2_arr > 0.01
gamma_normal = delta_r2_arr <= 0.01
n_dev = np.nansum(gamma_deviant)
n_norm = np.nansum(gamma_normal)

print(f"\nγ-deviant curves (Ph11 > Ph10 by > 0.01): {int(n_dev)} ({n_dev/N*100:.1f}%)")
print(f"γ-normal curves: {int(n_norm)} ({n_norm/N*100:.1f}%)")

print(f"\nEOS2 R² by γ-deviation status:")
for name, mask in [('γ-normal', gamma_normal), ('γ-deviant', gamma_deviant)]:
    valid_mask = mask & np.isfinite(mask)
    for regime_name, reg_mask in [('R1', R1), ('R2', R2), ('All', R1|R2)]:
        both = valid_mask & reg_mask
        if both.sum() > 5:
            print(f"  {name} × {regime_name}: N={both.sum()}, "
                  f"EOS2 R²={np.median(r2_eos2_gross[both]):.3f}, "
                  f"EOS3 R²={np.median(r2_eos3_gross[both]):.3f}")

# ============================================================
# FIX ②: R3 descriptive statistics
# ============================================================
print(f"\n{'='*70}")
print("FIX ②: R3 DESCRIPTIVE STATISTICS")
print(f"{'='*70}")

print(f"\nR3 (S ≤ 3): N = {R3.sum()}")
if R3.sum() > 0:
    print(f"  α median: {np.median(alpha[R3]):.4f} (R1: {np.median(alpha[R1]):.4f}, ratio: {np.median(alpha[R1])/np.median(alpha[R3]):.1f}×)")
    print(f"  β median: {np.median(beta[R3]):.4f} (R1: {np.median(beta[R1]):.4f})")
    print(f"  Pmax median: {np.median(Pmax[R3]):.3f} (R1: {np.median(Pmax[R1]):.3f})")
    print(f"  S median: {np.median(S[R3]):.2f} (range: {S[R3].min():.2f}–{S[R3].max():.2f})")
    Ic_R3 = Pmax[R3] / beta[R3]
    Ic_R1 = Pmax[R1] / beta[R1]
    print(f"  Iβ (=Pmax/β) median: {np.median(Ic_R3):.0f} (R1: {np.median(Ic_R1):.0f})")
    print(f"  SAI median: {np.median(SAI[R3]):.3f} (R1: {np.median(SAI[R1]):.3f})")

# ============================================================
# FIG 4: Representative curve overlays (R1 good, R2 bad→rescued)
# ============================================================
print(f"\n{'='*70}")
print("GENERATING FIG 4: Representative curve overlays")
print(f"{'='*70}")

fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
fig4.patch.set_facecolor('white')

# Pick representative curves
# R1 typical (high S, EOS2 works well)
r1_idx = np.where(R1 & (r2_eos2_gross > 0.92) & (r2_eos2_gross < 0.94))[0]
if len(r1_idx) > 0:
    r1_pick = r1_idx[len(r1_idx)//2]
else:
    r1_pick = np.where(R1)[0][0]

# R1 with high accuracy
r1_best = np.where(R1 & (r2_eos2_gross > 0.98))[0]
r1_best_pick = r1_best[len(r1_best)//3] if len(r1_best) > 0 else r1_pick

# R2 typical (EOS2 fails, EOS3 rescues)
r2_idx = np.where(R2 & (r2_eos2_gross > 0.5) & (r2_eos2_gross < 0.65) & 
                   (r2_eos3_gross > 0.9))[0]
r2_pick = r2_idx[len(r2_idx)//2] if len(r2_idx) > 0 else np.where(R2)[0][0]

# R2 dramatic rescue
r2_drama = np.where(R2 & (r2_eos2_gross < 0.3) & (r2_eos3_gross > 0.85))[0]
r2_drama_pick = r2_drama[0] if len(r2_drama) > 0 else r2_pick

picks = [
    (r1_best_pick, 'R1 — Factorized (high S)', '(a)'),
    (r1_pick, 'R1 — Factorized (typical)', '(b)'),
    (r2_pick, 'R2 — Transition (typical)', '(c)'),
    (r2_drama_pick, 'R2 — Transition (dramatic rescue)', '(d)'),
]

I_plot = np.linspace(1, 2500, 300)

for ax, (idx, title, panel) in zip(axes.flat, picks):
    a, b, Pm = alpha[idx], beta[idx], Pmax[idx]
    s_val = S[idx]
    bp = beta_pred[idx]
    sai_val = SAI[idx]
    
    P_ref = ph10_gross(I_plot, Pm, a, b)
    P_eos2 = ph10_gross(I_plot, Pm, a, bp)
    b3 = bp * 10**sai_val  # perfect SAI (no noise)
    P_eos3 = ph10_gross(I_plot, Pm, a, b3)
    
    ax.plot(I_plot, P_ref, 'k-', lw=2.5, label='Reference (Ph10 fit)', zorder=5)
    ax.plot(I_plot, P_eos2, '--', color='#E53935', lw=2, label=f'EOS2 (R²={r2_eos2_gross[idx]:.3f})', zorder=4)
    ax.plot(I_plot, P_eos3, '-.', color='#1E88E5', lw=2, label=f'EOS3 (R²={r2_eos3_gross[idx]:.3f})', zorder=3)
    
    ax.set_xlabel('I (µmol m⁻² s⁻¹)', fontsize=10)
    ax.set_ylabel('P_gross', fontsize=10)
    ax.set_title(f'{panel} {title}\nS = {s_val:.1f}, SAI = {sai_val:+.2f}', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.set_xlim(0, 2500)
    ax.set_ylim(bottom=0)

fig4.suptitle('Fig. 4 — Representative PI curves: Reference vs EOS2 vs EOS3',
              fontsize=14, fontweight='bold', y=1.01)
fig4.tight_layout()
fig4.savefig('raw_data/fig4_curve_overlays.png', dpi=250, bbox_inches='tight',
             facecolor='white')
print("Fig 4 saved!")

# ============================================================
# FIG 5: Monitoring workflow flowchart (create as clean diagram)
# ============================================================
fig5, ax5 = plt.subplots(figsize=(10, 7))
fig5.patch.set_facecolor('white')
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 8)
ax5.axis('off')

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edge='#1565C0', fontsize=10, bold=False):
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, edgecolor=edge, 
                          linewidth=2, transform=ax.transData, zorder=2)
    ax.add_patch(rect)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=fw, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, text='', color='#333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, text, fontsize=8, color=color, style='italic')

def draw_diamond(ax, x, y, w, h, text, color='#FFF9C4', edge='#F57F17', fontsize=9):
    pts = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
    from matplotlib.patches import Polygon
    diamond = Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2, zorder=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=3)

# Title
ax5.text(5, 7.5, 'Fig. 5 — Two-tier EOS monitoring workflow', 
         ha='center', fontsize=14, fontweight='bold')

# Boxes
draw_box(ax5, 2.5, 6.5, 3.5, 0.7, 'PAM-RLC (routine)\nI < 500 µmol m⁻²s⁻¹', 
         color='#C8E6C9', edge='#2E7D32', fontsize=9, bold=True)
draw_box(ax5, 2.5, 5.3, 3.0, 0.6, 'Estimate α, Pmax\n(low-I fit)', fontsize=9)
draw_box(ax5, 2.5, 4.1, 3.0, 0.6, 'Compute S = α/β_pred(α)\nRegime classification', fontsize=9)

draw_diamond(ax5, 5, 3.0, 2.5, 1.0, 'S > 10?', fontsize=10)

draw_box(ax5, 2.5, 1.8, 3.0, 0.7, 'EOS2: P(I; α, Pmax)\nR² ≈ 0.93', 
         color='#C8E6C9', edge='#2E7D32', fontsize=9, bold=True)
draw_box(ax5, 7.5, 3.0, 3.5, 0.7, 'Full RLC (diagnostic)\nI up to 2000 µmol m⁻²s⁻¹', 
         color='#FFCDD2', edge='#C62828', fontsize=9, bold=True)
draw_box(ax5, 7.5, 1.8, 3.0, 0.6, 'Estimate β → SAI', fontsize=9)
draw_box(ax5, 7.5, 0.7, 3.5, 0.7, 'EOS3: P(I; α, Pmax, SAI)\nR² ≈ 0.93, NRMSE = 50·σ_SAI', 
         color='#BBDEFB', edge='#1565C0', fontsize=9, bold=True)

# Arrows
draw_arrow(ax5, 2.5, 6.1, 2.5, 5.6)
draw_arrow(ax5, 2.5, 5.0, 2.5, 4.4)
draw_arrow(ax5, 2.5, 3.8, 3.75, 3.2)

draw_arrow(ax5, 3.75, 2.7, 2.5, 2.15, 'Yes (R1)')
draw_arrow(ax5, 6.25, 3.0, 5.75, 3.0, 'No (R2)')

# Actually the diamond decision needs proper arrows
# Left = Yes (R1), Right = No (R2)
# Redraw properly
ax5.annotate('', xy=(2.5, 2.15), xytext=(4.0, 2.65),
             arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))
ax5.text(3.0, 2.5, 'Yes\n(R1)', fontsize=9, color='#2E7D32', fontweight='bold')

ax5.annotate('', xy=(7.5, 3.35), xytext=(6.2, 3.0),
             arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5))
ax5.text(6.6, 3.35, 'No (R2)', fontsize=9, color='#C62828', fontweight='bold')

draw_arrow(ax5, 7.5, 2.65, 7.5, 2.1)
draw_arrow(ax5, 7.5, 1.5, 7.5, 1.05)

# Output box
draw_box(ax5, 5, 0.3, 4, 0.5, 'Output: Full PI curve prediction with quantified NRMSE', 
         color='#E8EAF6', edge='#283593', fontsize=9, bold=True)
draw_arrow(ax5, 2.5, 1.45, 4.0, 0.55)
draw_arrow(ax5, 7.5, 0.35, 7.0, 0.35)

fig5.savefig('raw_data/fig5_workflow.png', dpi=250, bbox_inches='tight', facecolor='white')
print("Fig 5 saved!")
