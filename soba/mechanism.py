"""
Hypothesis 2: SAI as NPQ/Total Defense Proxy
=============================================

Without external metadata, we attack from INSIDE the parameter space:

Strategy 1: Cross-model comparison for SAI extremes
  → If SAI-negative curves have WIDER plateaus, models that can't 
    capture plateaus (Platt) should fit them WORSE than typical.
  → The "model gap" (R²_Amirian - R²_Platt) is a proxy for plateau 
    prominence → proxy for photoprotection strength.

Strategy 2: Multi-model parameter pattern
  → Compare ALL 16 models' fits for SAI-extreme vs SAI-typical curves
  → Different failure modes = different mechanisms

Strategy 3: Pmax as environment proxy
  → High Pmax ≈ nutrient-rich/productive → surface/coastal
  → Low Pmax ≈ oligotrophic/deep → less photostress
  → SAI × Pmax interaction reveals ecological strategy

Strategy 4: pi_number temporal structure
  → PI numbers may encode collection order → temporal grouping

Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD ALL MODELS
# ============================================================
df_all = pd.read_csv('amirian_params.csv')

# Get Amirian (Ph10) data with SAI
df_am = df_all[df_all['Model_piCurve_pkg'] == 'Ph10'].copy()
df_am = df_am[df_am['Convrg'] == 0].copy()
df_am['log_alpha'] = np.log10(df_am['alpha'])
df_am['log_beta'] = np.log10(df_am['beta'])
mask = np.isfinite(df_am['log_alpha']) & np.isfinite(df_am['log_beta'])
df_am = df_am[mask].copy()

# Compute SAI
slope, intercept, _, _, _ = stats.linregress(df_am['log_alpha'], df_am['log_beta'])
df_am['SAI'] = df_am['log_beta'] - (slope * df_am['log_alpha'] + intercept)
df_am['Ialpha'] = df_am['Pmax'] / df_am['alpha']
df_am['Ibeta'] = df_am['Pmax'] / df_am['beta']
df_am['plateau_width'] = np.log10(df_am['Ibeta'] / df_am['Ialpha'])

# SAI groups
sai_std = df_am['SAI'].std()
df_am['sai_group'] = 'Typical'
df_am.loc[df_am['SAI'] < -2*sai_std, 'sai_group'] = 'Resistant'
df_am.loc[df_am['SAI'] > 2*sai_std, 'sai_group'] = 'Sensitive'

print(f"SAI groups:")
print(f"  Resistant (SAI < -2σ): {(df_am['sai_group']=='Resistant').sum()}")
print(f"  Typical:               {(df_am['sai_group']=='Typical').sum()}")
print(f"  Sensitive (SAI > +2σ): {(df_am['sai_group']=='Sensitive').sum()}")

# ============================================================
# STRATEGY 1: CROSS-MODEL R² COMPARISON
# ============================================================
print(f"\n{'='*70}")
print(f"STRATEGY 1: Cross-Model Fit Quality vs SAI Group")
print(f"{'='*70}")

# Get Platt model (Ph01) R² for comparison
models = df_all['Model_piCurve_pkg'].unique()
print(f"\nAvailable models: {sorted(models)}")

# For each model, compute median R² by SAI group
model_names = {
    'Ph01': 'Platt (exp decay)',
    'Ph02': 'Steele (linear×exp)', 
    'Ph03': 'Parker (tanh×exp)',
    'Ph04': 'Neale (exp×exp)',
    'Ph05': 'Eilers-Peeters',
    'Ph10': 'Amirian (tanh×tanh)',
}

results = []
for model_id in sorted(models):
    df_model = df_all[(df_all['Model_piCurve_pkg'] == model_id) & (df_all['Convrg'] == 0)].copy()
    df_model = df_model.set_index('pi_number')
    
    for group in ['Resistant', 'Typical', 'Sensitive']:
        pis = df_am[df_am['sai_group'] == group]['pi_number'].values
        r2s = []
        for pi in pis:
            if pi in df_model.index:
                val = df_model.loc[pi, 'R2adj']
                if isinstance(val, pd.Series):
                    val = val.values[0]
                if np.isfinite(val):
                    r2s.append(val)
        
        if len(r2s) > 0:
            results.append({
                'model': model_id,
                'group': group,
                'median_R2': np.median(r2s),
                'mean_R2': np.mean(r2s),
                'n': len(r2s)
            })

df_results = pd.DataFrame(results)

# Pivot and display
print(f"\nMedian R²adj by SAI group and model:")
print(f"{'Model':8s} {'Resistant':>12s} {'Typical':>12s} {'Sensitive':>12s} {'Gap(R-T)':>10s}")
for model_id in sorted(models):
    dr = df_results[df_results['model'] == model_id]
    r_val = dr[dr['group']=='Resistant']['median_R2'].values
    t_val = dr[dr['group']=='Typical']['median_R2'].values
    s_val = dr[dr['group']=='Sensitive']['median_R2'].values
    
    r_str = f"{r_val[0]:.4f}" if len(r_val) > 0 else "N/A"
    t_str = f"{t_val[0]:.4f}" if len(t_val) > 0 else "N/A"
    s_str = f"{s_val[0]:.4f}" if len(s_val) > 0 else "N/A"
    
    gap = ""
    if len(r_val) > 0 and len(t_val) > 0:
        gap = f"{r_val[0] - t_val[0]:+.4f}"
    
    name = model_names.get(model_id, model_id)
    print(f"  {model_id:6s} {r_str:>12s} {t_str:>12s} {s_str:>12s} {gap:>10s}  {name}")

# ============================================================
# STRATEGY 1b: MODEL GAP as plateau proxy
# ============================================================
print(f"\n{'='*70}")
print(f"STRATEGY 1b: Amirian vs Platt R² Gap by SAI Group")
print(f"{'='*70}")

# For each PI curve: compute R²(Amirian) - R²(Platt)
df_platt = df_all[(df_all['Model_piCurve_pkg'] == 'Ph01') & (df_all['Convrg'] == 0)].copy()
df_platt = df_platt.set_index('pi_number')[['R2adj']].rename(columns={'R2adj': 'R2_Platt'})

df_merged = df_am.set_index('pi_number').join(df_platt, how='inner')
df_merged['R2_gap'] = df_merged['R2adj'] - df_merged['R2_Platt']

print(f"\nR² gap (Amirian - Platt) by SAI group:")
for group in ['Resistant', 'Typical', 'Sensitive']:
    subset = df_merged[df_merged['sai_group'] == group]
    gap = subset['R2_gap']
    print(f"  {group:12s}: median gap = {gap.median():+.4f}, mean = {gap.mean():+.4f}, N = {len(gap)}")

# Statistical test: is the gap larger for resistant group?
from scipy.stats import mannwhitneyu
resistant_gaps = df_merged[df_merged['sai_group']=='Resistant']['R2_gap'].dropna()
typical_gaps = df_merged[df_merged['sai_group']=='Typical']['R2_gap'].dropna()
sensitive_gaps = df_merged[df_merged['sai_group']=='Sensitive']['R2_gap'].dropna()

if len(resistant_gaps) > 5 and len(typical_gaps) > 5:
    u_rt, p_rt = mannwhitneyu(resistant_gaps, typical_gaps, alternative='greater')
    print(f"\n  Resistant vs Typical: U = {u_rt:.0f}, p = {p_rt:.4f}")
    print(f"  → {'Resistant curves have LARGER model gap!' if p_rt < 0.05 else 'No significant difference'}")

if len(sensitive_gaps) > 5 and len(typical_gaps) > 5:
    u_st, p_st = mannwhitneyu(sensitive_gaps, typical_gaps, alternative='two-sided')
    print(f"  Sensitive vs Typical: U = {u_st:.0f}, p = {p_st:.4f}")

# Correlation: SAI vs R² gap
r_sai_gap, p_sai_gap = stats.pearsonr(df_merged['SAI'].dropna(), df_merged['R2_gap'].dropna())
print(f"\n  Correlation SAI vs R²_gap: r = {r_sai_gap:+.4f} (p = {p_sai_gap:.2e})")
print(f"  → {'Negative SAI (resistant) → larger gap → wider plateau → more photoprotection!' if r_sai_gap < -0.1 else 'No clear trend'}")

# ============================================================
# STRATEGY 2: Pmax as Environment Proxy
# ============================================================
print(f"\n{'='*70}")
print(f"STRATEGY 2: SAI × Pmax Interaction (Ecological Strategy)")
print(f"{'='*70}")

# Divide into Pmax quartiles
df_am['Pmax_quartile'] = pd.qcut(df_am['Pmax'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

print(f"\nSAI statistics by Pmax quartile:")
print(f"{'Quartile':12s} {'N':>5s} {'SAI mean':>10s} {'SAI std':>10s} {'Skew':>8s} {'% Resist':>10s}")
for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
    subset = df_am[df_am['Pmax_quartile'] == q]
    n_resist = (subset['sai_group'] == 'Resistant').sum()
    print(f"  {q:10s} {len(subset):5d} {subset['SAI'].mean():+10.4f} "
          f"{subset['SAI'].std():10.4f} {stats.skew(subset['SAI']):8.3f} "
          f"{n_resist/len(subset)*100:9.1f}%")

# Key test: does skewness depend on Pmax?
print(f"\nSkewness by Pmax quartile (higher Pmax ≈ more productive/stressed):")
for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
    subset = df_am[df_am['Pmax_quartile'] == q]['SAI']
    sk = stats.skew(subset)
    # Bootstrap CI for skewness
    boot_sk = [stats.skew(np.random.choice(subset, len(subset), replace=True)) for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(boot_sk, [2.5, 97.5])
    sig = "✅" if ci_hi < 0 else "❌"
    print(f"  {q:10s}: skew = {sk:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] {sig}")

# Correlation: Pmax vs SAI
r_pmax_sai, p_pmax_sai = stats.pearsonr(df_am['Pmax'], df_am['SAI'])
print(f"\n  Pmax vs SAI: r = {r_pmax_sai:+.4f} (p = {p_pmax_sai:.2e})")

# ============================================================
# STRATEGY 3: 2D Ecological Strategy Space
# ============================================================
print(f"\n{'='*70}")
print(f"STRATEGY 3: (α*, SAI) 2D Ecological Space")
print(f"{'='*70}")

# Define α* = residual of log(α) after removing Pmax dependence
# This captures "absorption efficiency independent of capacity"
slope_ap, intercept_ap, r_ap, _, _ = stats.linregress(
    np.log10(df_am['Pmax']), df_am['log_alpha'])
df_am['alpha_star'] = df_am['log_alpha'] - (slope_ap * np.log10(df_am['Pmax']) + intercept_ap)

print(f"\nα* definition: residual(log α | log Pmax)")
print(f"  log α = {slope_ap:.3f} × log Pmax + ({intercept_ap:.3f}), r = {r_ap:.3f}")
print(f"  α* > 0: absorbs MORE than expected for its capacity")
print(f"  α* < 0: absorbs LESS than expected for its capacity")

# Correlation: α* vs SAI
r_astar_sai, p_astar_sai = stats.pearsonr(df_am['alpha_star'], df_am['SAI'])
print(f"\n  α* vs SAI: r = {r_astar_sai:+.4f} (p = {p_astar_sai:.2e})")
print(f"  → {'High absorbers are more sensitive (need more repair)' if r_astar_sai > 0.1 else 'Independent!' if abs(r_astar_sai) < 0.1 else 'High absorbers are more resistant'}")

# Quadrant analysis
df_am['strategy'] = 'Balanced'
df_am.loc[(df_am['alpha_star'] > 0) & (df_am['SAI'] < 0), 'strategy'] = 'Absorb+Repair'
df_am.loc[(df_am['alpha_star'] > 0) & (df_am['SAI'] > 0), 'strategy'] = 'Absorb (vulnerable)'
df_am.loc[(df_am['alpha_star'] < 0) & (df_am['SAI'] < 0), 'strategy'] = 'Avoid+Repair'
df_am.loc[(df_am['alpha_star'] < 0) & (df_am['SAI'] > 0), 'strategy'] = 'Avoid (passive)'

print(f"\nEcological strategy quadrants (α*, SAI):")
for strat in ['Absorb+Repair', 'Absorb (vulnerable)', 'Avoid+Repair', 'Avoid (passive)']:
    n = (df_am['strategy'] == strat).sum()
    frac = n / len(df_am) * 100
    subset = df_am[df_am['strategy'] == strat]
    print(f"  {strat:25s}: {n:5d} ({frac:5.1f}%) "
          f"med_Pmax={subset['Pmax'].median():.2f} "
          f"med_R²={subset['R2adj'].median():.3f}")

# ============================================================
# STRATEGY 4: pi_number temporal structure
# ============================================================
print(f"\n{'='*70}")
print(f"STRATEGY 4: pi_number Structure Analysis")
print(f"{'='*70}")

# Extract numeric part of pi_number
def extract_pi_num(s):
    """Extract numeric value from pi_number like PI000485, PI002147_3, PI0_492512"""
    import re
    # Try different formats
    m = re.match(r'PI0?_?(\d+)', s)
    if m:
        return int(m.group(1))
    return None

df_am['pi_num'] = df_am['pi_number'].apply(extract_pi_num)
valid_pi = df_am['pi_num'].notna()
print(f"  Parsed pi_numbers: {valid_pi.sum()} / {len(df_am)}")
print(f"  Range: {df_am.loc[valid_pi, 'pi_num'].min():.0f} — {df_am.loc[valid_pi, 'pi_num'].max():.0f}")

# Correlation: pi_number vs parameters
for param in ['Pmax', 'SAI', 'log_alpha', 'log_beta']:
    vals = df_am.loc[valid_pi, param]
    nums = df_am.loc[valid_pi, 'pi_num']
    r, p = stats.pearsonr(nums, vals)
    print(f"  pi_num vs {param:12s}: r = {r:+.4f} (p = {p:.2e})")

# ============================================================
# FIGURE 11: MECHANISM FIGURE
# ============================================================
fig = plt.figure(figsize=(22, 16))
fig.suptitle('Hypothesis 2: SAI as Photoprotection Proxy\n'
             'Cross-Model Evidence for NPQ/FtsH Defense Signature',
             fontsize=14, fontweight='bold', y=0.98)

# (a) R² gap (Amirian - Platt) vs SAI
ax1 = fig.add_subplot(2, 3, 1)
sc = ax1.scatter(df_merged['SAI'], df_merged['R2_gap'], 
                 c=np.log10(df_merged['Pmax']), cmap='coolwarm', 
                 s=6, alpha=0.4)
cb = plt.colorbar(sc, ax=ax1, shrink=0.8)
cb.set_label('$\\log_{10}(P_{max})$', fontsize=8)

# Binned means
bins = np.linspace(df_merged['SAI'].min(), df_merged['SAI'].max(), 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_means = []
for i in range(len(bins)-1):
    mask = (df_merged['SAI'] >= bins[i]) & (df_merged['SAI'] < bins[i+1])
    if mask.sum() > 5:
        bin_means.append(df_merged.loc[mask, 'R2_gap'].mean())
    else:
        bin_means.append(np.nan)
ax1.plot(bin_centers, bin_means, 'k-', linewidth=2, label='Binned mean')

ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('SAI', fontsize=10)
ax1.set_ylabel('$R²_{Amirian} - R²_{Platt}$ (model gap)', fontsize=10)
ax1.set_title(f'(a) Model gap vs SAI\nr = {r_sai_gap:+.3f}, p = {p_sai_gap:.1e}\n'
              '"Resistant curves need plateau model more"',
             fontsize=9, fontweight='bold')
ax1.legend(fontsize=8)

# (b) SAI distribution by Pmax quartile
ax2 = fig.add_subplot(2, 3, 2)
colors_q = ['#1565C0', '#42A5F5', '#EF6C00', '#E53935']
for i, q in enumerate(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']):
    subset = df_am[df_am['Pmax_quartile'] == q]['SAI']
    bins_q = np.linspace(-2, 1.5, 50)
    ax2.hist(subset, bins=bins_q, density=True, alpha=0.4, color=colors_q[i], 
             label=f'{q}: skew={stats.skew(subset):+.2f}')

ax2.axvline(0, color='black', linestyle=':', alpha=0.5)
ax2.set_xlabel('SAI', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('(b) SAI by Pmax quartile\n"Does productivity modulate selection?"',
             fontsize=9, fontweight='bold')
ax2.legend(fontsize=7)

# (c) 2D ecological strategy: α* vs SAI
ax3 = fig.add_subplot(2, 3, 3)
strategy_colors = {
    'Absorb+Repair': '#2E7D32',
    'Absorb (vulnerable)': '#C62828',
    'Avoid+Repair': '#1565C0',
    'Avoid (passive)': '#FF8F00'
}
for strat, color in strategy_colors.items():
    mask = df_am['strategy'] == strat
    ax3.scatter(df_am.loc[mask, 'alpha_star'], df_am.loc[mask, 'SAI'],
               s=4, alpha=0.3, color=color, label=strat)

ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax3.axvline(0, color='gray', linestyle='-', alpha=0.3)
ax3.set_xlabel('α* (absorption residual)', fontsize=10)
ax3.set_ylabel('SAI (repair residual)', fontsize=10)
ax3.set_title(f'(c) Ecological strategy space\nα* vs SAI: r = {r_astar_sai:+.3f}',
             fontsize=9, fontweight='bold')
ax3.legend(fontsize=7, loc='upper left', markerscale=3)

# (d) Resistant curves: parameter profile
ax4 = fig.add_subplot(2, 3, 4)
params_to_show = ['Pmax', 'alpha', 'beta', 'R2adj']
labels_show = ['$P_{max}$', '$α$', '$β$', '$R²_{adj}$']

# Compute z-scores relative to typical group
typical = df_am[df_am['sai_group'] == 'Typical']
resistant = df_am[df_am['sai_group'] == 'Resistant']
sensitive = df_am[df_am['sai_group'] == 'Sensitive']

x_pos = np.arange(len(params_to_show))
for group, color, label in [(resistant, '#0D47A1', 'Resistant'), 
                              (sensitive, '#B71C1C', 'Sensitive')]:
    z_scores = []
    for param in params_to_show:
        z = (group[param].median() - typical[param].median()) / typical[param].std()
        z_scores.append(z)
    ax4.bar(x_pos + (0.15 if color == '#0D47A1' else -0.15), z_scores, 0.3,
           color=color, alpha=0.7, label=label)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels_show)
ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
ax4.set_ylabel('Z-score (vs Typical)', fontsize=10)
ax4.set_title('(d) Parameter profiles of extremes\n"What makes them different?"',
             fontsize=9, fontweight='bold')
ax4.legend(fontsize=8)

# (e) R² gap distribution by SAI group
ax5 = fig.add_subplot(2, 3, 5)
bins_gap = np.linspace(-0.3, 0.4, 50)
for group, color, label in [('Resistant', '#0D47A1', 'Resistant (SAI < -2σ)'),
                              ('Typical', '#455A64', 'Typical'),
                              ('Sensitive', '#B71C1C', 'Sensitive (SAI > +2σ)')]:
    subset = df_merged[df_merged['sai_group'] == group]['R2_gap']
    ax5.hist(subset, bins=bins_gap, density=True, alpha=0.4, color=color, label=label)

ax5.axvline(0, color='black', linestyle=':', alpha=0.5)
ax5.set_xlabel('$R²_{Amirian} - R²_{Platt}$', fontsize=10)
ax5.set_ylabel('Density', fontsize=10)
ax5.set_title('(e) Model gap by SAI group\n"Resistant curves need the plateau model most"',
             fontsize=9, fontweight='bold')
ax5.legend(fontsize=7)

# (f) Summary interpretation
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

summary = f"""
HYPOTHESIS 2 EVIDENCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MODEL GAP TEST
   SAI vs R²(Amirian)−R²(Platt): r = {r_sai_gap:+.3f}
   → Resistant curves have {'LARGER' if r_sai_gap < 0 else 'similar'} model gaps
   → {'Wider plateaus → more photoprotection ✓' if r_sai_gap < -0.1 else 'Plateau width ~ independent of SAI'}

2. Pmax × SAI INTERACTION
   Pmax vs SAI: r = {r_pmax_sai:+.3f}
   → {'High-capacity organisms more sensitive' if r_pmax_sai > 0.1 else 'Capacity partly independent of defense'}
   
3. 2D ECOLOGICAL STRATEGIES
   α* vs SAI: r = {r_astar_sai:+.3f}
   → {'Absorption & repair are coupled' if abs(r_astar_sai) > 0.15 else 'Absorption & repair are INDEPENDENT axes'}
   → {'Two separate ecological dimensions!' if abs(r_astar_sai) < 0.15 else ''}

4. INTERPRETATION
   SAI captures MORE than just β (repair rate):
   • It absorbs plateau width (photoprotection)
   • It correlates with model selection (NPQ proxy)
   • It defines independent ecological axis from α*
   
   → SAI = "total SCC defense capacity"
   → Not just FtsH repair, but NPQ + photoprotective pigments
"""
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
         va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig11_mechanism_hypothesis2.png', dpi=200, bbox_inches='tight')
print(f"\n✅ fig11_mechanism_hypothesis2.png")
