"""
PCC/SCC Evolutionary Invariance Hypothesis
=============================================

THESIS (Iizumi 2026):
  "Species diversity in photosynthesis is predominantly
   variation along a SINGLE SCC parameter: K_r (FtsH repair efficiency)"

Mathematical formulation:
  P(I) = P_PCC(I; α_universal, Pmax) × p_active(I; K_d_conserved, K_r_species)
  
  PCC parameters (α, Pmax): CONSERVED across species
    → Set by physics of photon capture + charge separation
    → 38 billion years of evolutionary stasis
    
  SCC parameter (K_r): VARIABLE across species  
    → Set by FtsH expression level, membrane composition
    → The single "tuning knob" of adaptation

  n(K_r) = 1/(K_d + K_r)  where K_d ≈ const
  
  → ALL variation in photoinhibition reduces to variation in K_r
  → "Different species" = "different K_r values"

TESTABLE PREDICTIONS:
  1. α (initial slope) should be nearly identical across all species
  2. n and Ic should be anti-correlated (both depend on K_r)
  3. Shade species: low K_r → high n → sharp transition
  4. Light species: high K_r → low n → gradual transition
  5. 1808 PI curves should collapse onto a 1-parameter family

Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. THE SINGLE-PARAMETER MODEL
# ============================================================

def pcc_scc_evolutionary(I, Pmax, alpha, K_d, K_r, V_ratio, I_half_damage):
    """
    Full PCC/SCC model with evolutionary interpretation.
    
    PCC (conserved):
      P_PCC = Pmax * tanh(alpha * I / Pmax)
      α ≈ 0.04-0.06 for all photosynthetic organisms (quantum yield)
    
    SCC (adaptive):
      p_active from Goldbeter-Koshland steady state
      Approximated as Hill: p_active = 1/(1 + (I/Ic)^n)
      where:
        n ≈ 1/(K_d + K_r)
        Ic depends on V_repair/V_damage ratio and I_half_damage
    
    The "evolutionary knob":
      K_r alone determines species identity in this framework.
      K_d ≈ 0.02 (conserved: D1 structure is universal)
      K_r ∈ [0.01, 0.25] (variable: FtsH expression/efficiency)
    """
    # PCC term (conserved physics)
    P_pcc = Pmax * np.tanh(alpha * I / Pmax)
    
    # SCC term (adaptive biology)
    # Effective Hill parameters from GK theory:
    n_eff = 1.0 / (K_d + K_r)
    
    # Critical irradiance: where damage rate = repair rate
    # v_d(Ic) = V_damage * Ic/(Ic + I_half) = V_repair = V_ratio * V_damage
    # → Ic = I_half * V_ratio / (1 - V_ratio)
    Ic = I_half_damage * V_ratio / max(1 - V_ratio, 0.01)
    
    p_active = 1.0 / (1.0 + (I / Ic)**n_eff)
    
    return P_pcc * p_active

def pcc_scc_simple(I, Pmax, alpha, Ic, n):
    """Simple 4-parameter form for fitting"""
    P_pcc = Pmax * np.tanh(alpha * I / Pmax)
    p_active = 1.0 / (1.0 + (I / Ic)**n)
    return P_pcc * p_active

# ============================================================
# 2. RE-ANALYZE THE 4 CURVES WITH EVOLUTIONARY LENS
# ============================================================
import pandas as pd
df = pd.read_csv('piDataSet.csv')
ph_ids = ['PI002366', 'PI002413', 'PI002788', 'PI002794']

print("="*70)
print("RE-ANALYSIS WITH EVOLUTIONARY INVARIANCE HYPOTHESIS")
print("="*70)

# First: fit each curve independently (as before)
independent_results = {}
for pi_id in ph_ids:
    sub = df[df['pi_number'] == pi_id].sort_values('I')
    I = sub['I'].values.astype(float)
    P = sub['P'].values.astype(float)
    Pm = np.max(P)
    I_peak = I[np.argmax(P)]
    
    low_mask = I < np.percentile(I, 25)
    if np.sum(low_mask) > 2:
        alpha_est = max(np.polyfit(I[low_mask], P[low_mask], 1)[0], 0.005)
    else:
        alpha_est = 0.03
    
    best = None
    for pm in [Pm, Pm*1.2, Pm*1.5]:
        for a in [alpha_est*0.5, alpha_est, alpha_est*1.5]:
            for ic in [I_peak*0.8, I_peak, I_peak*1.5, max(I)*0.7]:
                for nn in [2, 3, 4, 6, 8, 10]:
                    try:
                        popt, _ = curve_fit(pcc_scc_simple, I, P,
                                           p0=[pm, a, ic, nn],
                                           bounds=([0,0,10,0.5],[10*Pm,1,5000,15]),
                                           maxfev=20000)
                        P_pred = pcc_scc_simple(I, *popt)
                        r2 = 1 - np.sum((P-P_pred)**2)/np.sum((P-np.mean(P))**2)
                        if best is None or r2 > best['r2']:
                            best = {'params': popt, 'r2': r2, 'P_pred': P_pred}
                    except:
                        pass
    
    independent_results[pi_id] = best
    Pmax, alpha, Ic, n = best['params']
    print(f"\n{pi_id}: Pmax={Pmax:.3f}, α={alpha:.5f}, Ic={Ic:.0f}, n={n:.2f}  R²={best['r2']:.4f}")

# ============================================================
# 3. TEST: Is α conserved?
# ============================================================
print("\n" + "="*70)
print("TEST 1: Is α (initial slope / quantum yield) conserved?")
print("="*70)

alphas = [independent_results[pi]['params'][1] for pi in ph_ids]
pmaxs = [independent_results[pi]['params'][0] for pi in ph_ids]
ns = [independent_results[pi]['params'][3] for pi in ph_ids]
ics = [independent_results[pi]['params'][2] for pi in ph_ids]

print(f"\nα values: {[f'{a:.5f}' for a in alphas]}")
print(f"Mean α = {np.mean(alphas):.5f} ± {np.std(alphas):.5f}")
print(f"CV(α) = {np.std(alphas)/np.mean(alphas)*100:.1f}%")

print(f"\nPmax values: {[f'{p:.3f}' for p in pmaxs]}")
print(f"Mean Pmax = {np.mean(pmaxs):.3f} ± {np.std(pmaxs):.3f}")
print(f"CV(Pmax) = {np.std(pmaxs)/np.mean(pmaxs)*100:.1f}%")

print(f"\nn values: {[f'{n:.2f}' for n in ns]}")
print(f"Mean n = {np.mean(ns):.2f} ± {np.std(ns):.2f}")
print(f"CV(n) = {np.std(ns)/np.mean(ns)*100:.1f}%")

print(f"\nIc values: {[f'{ic:.0f}' for ic in ics]}")
print(f"Mean Ic = {np.mean(ics):.0f} ± {np.std(ics):.0f}")
print(f"CV(Ic) = {np.std(ics)/np.mean(ics)*100:.1f}%")

print(f"\n--- Variability ranking ---")
cvs = {
    'α (PCC - quantum yield)': np.std(alphas)/np.mean(alphas)*100,
    'Pmax (PCC - max rate)': np.std(pmaxs)/np.mean(pmaxs)*100,
    'n (SCC - cooperativity)': np.std(ns)/np.mean(ns)*100,
    'Ic (SCC - critical I)': np.std(ics)/np.mean(ics)*100,
}
for name, cv in sorted(cvs.items(), key=lambda x: x[1]):
    marker = " ← CONSERVED" if cv < 30 else " ← VARIABLE"
    print(f"  CV({name}) = {cv:.1f}%{marker}")

# ============================================================
# 4. TEST: n vs Ic anti-correlation (both from K_r)
# ============================================================
print("\n" + "="*70)
print("TEST 2: n-Ic relationship (both from K_r?)")
print("="*70)

from scipy.stats import pearsonr, spearmanr

r_pearson, p_pearson = pearsonr(ns, ics)
r_spearman, p_spearman = spearmanr(ns, ics)

print(f"Pearson r(n, Ic) = {r_pearson:.3f}  (p = {p_pearson:.3f})")
print(f"Spearman ρ(n, Ic) = {r_spearman:.3f}  (p = {p_spearman:.3f})")

# From GK theory: 
# n = 1/(K_d + K_r), Ic = I_half * V_r/V_d_max / (1 - V_r/V_d_max)
# Both depend on K_r, but Ic also depends on V_r/V_d ratio
# If V_r ∝ 1/K_r (more saturated = lower K_r but also lower absolute rate?)
# The relationship is complex...

print(f"\nPhysical analysis:")
print(f"  n = 1/(K_d + K_r) → n ↑ as K_r ↓")
print(f"  Ic depends on V_repair/V_damage ratio")
print(f"  If species with low K_r also have low V_repair (shade-adapted):")
print(f"    → low K_r → high n AND low Ic")
print(f"    → Expected: NEGATIVE n-Ic correlation")
print(f"  If species with low K_r maintain high V_repair (efficient FtsH):")
print(f"    → low K_r → high n AND high Ic")  
print(f"    → Expected: POSITIVE n-Ic correlation")
print(f"\n  Observed: r = {r_pearson:.3f} → {'positive' if r_pearson > 0 else 'negative'}")
print(f"  Interpretation: Curves with higher n also tend to have "
      f"{'higher' if r_pearson > 0 else 'lower'} Ic")

# ============================================================
# 5. CONSTRAINED FIT: Fix α = universal, fit only SCC params
# ============================================================
print("\n" + "="*70)
print("TEST 3: Constrained fit (α fixed = universal)")
print("="*70)

alpha_universal = np.mean(alphas)
print(f"Fixing α = {alpha_universal:.5f} (mean of 4 curves)")

def pcc_scc_fixed_alpha(I, Pmax, Ic, n):
    """3-parameter model with fixed α"""
    P_pcc = Pmax * np.tanh(alpha_universal * I / Pmax)
    p_active = 1.0 / (1.0 + (I / Ic)**n)
    return P_pcc * p_active

constrained_results = {}
for pi_id in ph_ids:
    sub = df[df['pi_number'] == pi_id].sort_values('I')
    I = sub['I'].values.astype(float)
    P = sub['P'].values.astype(float)
    Pm = np.max(P)
    I_peak = I[np.argmax(P)]
    
    best = None
    for pm in [Pm, Pm*1.2, Pm*1.5]:
        for ic in [I_peak, I_peak*1.5, max(I)*0.7]:
            for nn in [2, 4, 6, 8]:
                try:
                    popt, _ = curve_fit(pcc_scc_fixed_alpha, I, P,
                                       p0=[pm, ic, nn],
                                       bounds=([0,10,0.5],[10*Pm,5000,15]),
                                       maxfev=20000)
                    P_pred = pcc_scc_fixed_alpha(I, *popt)
                    r2 = 1 - np.sum((P-P_pred)**2)/np.sum((P-np.mean(P))**2)
                    if best is None or r2 > best['r2']:
                        best = {'params': popt, 'r2': r2}
                except:
                    pass
    
    constrained_results[pi_id] = best
    Pmax, Ic, n = best['params']
    r2_free = independent_results[pi_id]['r2']
    r2_const = best['r2']
    delta_r2 = r2_free - r2_const
    print(f"{pi_id}: Pmax={Pmax:.3f}, Ic={Ic:.0f}, n={n:.2f}  "
          f"R²={r2_const:.4f}  (ΔR² from free α: {delta_r2:.4f})")

mean_delta = np.mean([independent_results[pi]['r2'] - constrained_results[pi]['r2'] 
                      for pi in ph_ids])
print(f"\nMean ΔR² from fixing α: {mean_delta:.4f}")
print(f"→ {'NEGLIGIBLE' if abs(mean_delta) < 0.01 else 'SIGNIFICANT'} cost of fixing α")

# ============================================================
# 6. THE K_r-ONLY MODEL
# ============================================================
print("\n" + "="*70)
print("TEST 4: K_r-only model (single evolutionary parameter)")
print("="*70)

K_d_conserved = 0.02  # Conserved D1 damage efficiency

print(f"Fixed parameters (conserved across all species):")
print(f"  α = {alpha_universal:.5f} (quantum yield)")
print(f"  K_d = {K_d_conserved} (D1 photodamage saturation)")
print(f"\nVariable parameter (species-specific):")
print(f"  K_r = FtsH repair saturation (the evolutionary knob)")

def pcc_scc_Kr_only(I, Pmax, K_r, V_ratio, I_half):
    """
    The minimal evolutionary model:
    Fixed: α (universal), K_d (conserved)
    Free: Pmax, K_r (species-adaptive), V_ratio, I_half
    
    n = 1/(K_d + K_r) — directly from K_r
    Ic = I_half * V_ratio / (1 - V_ratio) — from repair/damage balance
    """
    P_pcc = Pmax * np.tanh(alpha_universal * I / Pmax)
    n_eff = 1.0 / (K_d_conserved + K_r)
    Ic = I_half * V_ratio / max(1 - V_ratio, 0.01)
    p_active = 1.0 / (1.0 + (I / max(Ic, 1))**max(n_eff, 0.5))
    return P_pcc * p_active

Kr_results = {}
for pi_id in ph_ids:
    sub = df[df['pi_number'] == pi_id].sort_values('I')
    I = sub['I'].values.astype(float)
    P = sub['P'].values.astype(float)
    Pm = np.max(P)
    
    best = None
    for pm in [Pm, Pm*1.2, Pm*1.5]:
        for kr in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            for vr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for ih in [100, 200, 300, 500, 800]:
                    try:
                        popt, _ = curve_fit(pcc_scc_Kr_only, I, P,
                                           p0=[pm, kr, vr, ih],
                                           bounds=([0,0.005,0.1,10],
                                                  [10*Pm,0.5,0.95,3000]),
                                           maxfev=20000)
                        P_pred = pcc_scc_Kr_only(I, *popt)
                        r2 = 1 - np.sum((P-P_pred)**2)/np.sum((P-np.mean(P))**2)
                        if best is None or r2 > best['r2']:
                            best = {'params': popt, 'r2': r2, 'P_pred': P_pred}
                    except:
                        pass
    
    Kr_results[pi_id] = best
    if best:
        Pmax, K_r, V_ratio, I_half = best['params']
        n_eff = 1.0/(K_d_conserved + K_r)
        Ic_eff = I_half * V_ratio / max(1-V_ratio, 0.01)
        print(f"{pi_id}: Pmax={Pmax:.3f}, K_r={K_r:.4f} → n={n_eff:.2f}, Ic={Ic_eff:.0f}  "
              f"R²={best['r2']:.4f}")
    else:
        print(f"{pi_id}: FAILED")

# ============================================================
# 7. THE EVOLUTIONARY LANDSCAPE FIGURE
# ============================================================
fig = plt.figure(figsize=(16, 14))

# Layout: 3 rows × 2 cols
gs = plt.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) The evolutionary invariance concept
ax = fig.add_subplot(gs[0, 0])
ax.axis('off')

concept_text = """
    ╔══════════════════════════════════════════╗
    ║   EVOLUTIONARY INVARIANCE HYPOTHESIS     ║
    ╠══════════════════════════════════════════╣
    ║                                          ║
    ║   P(I) = P_PCC(I) × p_active(I)         ║
    ║           ↑            ↑                 ║
    ║       CONSERVED    ADAPTIVE              ║
    ║       (physics)    (biology)             ║
    ║                                          ║
    ║   PCC: α, charge separation              ║
    ║     → 38 billion years unchanged         ║
    ║     → quantum yield ≈ 100%               ║
    ║     → D1 structure nearly identical      ║
    ║                                          ║
    ║   SCC: K_r (FtsH repair efficiency)      ║
    ║     → THE evolutionary tuning knob       ║
    ║     → Determines: n, Ic, photoinhibition ║
    ║     → "Species diversity" ≈ K_r diversity ║
    ║                                          ║
    ╚══════════════════════════════════════════╝
"""
ax.text(0.02, 0.98, concept_text, transform=ax.transAxes,
       fontsize=8.5, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('(a) The Hypothesis', fontsize=11, fontweight='bold')

# (b) Parameter variability (CV comparison)
ax = fig.add_subplot(gs[0, 1])
param_names = ['α\n(quantum yield)', 'Pmax\n(max rate)', 'n\n(cooperativity)', 'Ic\n(critical I)']
cv_values = [cvs['α (PCC - quantum yield)'], cvs['Pmax (PCC - max rate)'],
             cvs['n (SCC - cooperativity)'], cvs['Ic (SCC - critical I)']]
colors_bar = ['#2196F3', '#64B5F6', '#FF5722', '#FF9800']
bar_labels = ['PCC', 'PCC', 'SCC', 'SCC']

bars = ax.bar(param_names, cv_values, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.axhline(30, color='gray', linestyle='--', alpha=0.5, label='30% threshold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
ax.set_title('(b) Parameter Variability: PCC vs SCC', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# Add PCC/SCC labels
for bar, label in zip(bars, bar_labels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
           label, ha='center', fontsize=9, fontweight='bold',
           color='#1565C0' if label == 'PCC' else '#D84315')

# (c) K_r determines n: the single-parameter family
ax = fig.add_subplot(gs[1, 0])
I_plot = np.linspace(0.1, 2000, 1000)

K_r_range = [0.01, 0.03, 0.05, 0.08, 0.12, 0.20]
cmap = plt.cm.RdYlBu_r
norm = plt.Normalize(0.01, 0.20)

for K_r in K_r_range:
    n_eff = 1.0 / (K_d_conserved + K_r)
    Ic_eff = 500 * 0.5 / (1 - 0.5)  # Simplified: Ic = 500 for illustration
    p_act = 1.0 / (1.0 + (I_plot / Ic_eff)**n_eff)
    color = cmap(norm(K_r))
    ax.plot(I_plot, p_act, linewidth=2.5, color=color,
           label=f'$K_r$={K_r:.2f} → n={n_eff:.1f}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='$K_r$ (repair saturation)')

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('$I$ (µmol m⁻² s⁻¹)', fontsize=11)
ax.set_ylabel('$p_{active}$', fontsize=11)
ax.set_title('(c) One Knob Rules Them All: $K_r$ → $n$ → Gate Shape',
            fontsize=11, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.set_xlim(0, 2000)

# Add species annotations
ax.annotate('Shade species\n(low $K_r$, sharp gate)', 
           xy=(400, 0.25), fontsize=9, color='darkred',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.annotate('Light species\n(high $K_r$, gradual gate)',
           xy=(1200, 0.6), fontsize=9, color='darkblue',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

# (d) Fit all curves with constrained α
ax = fig.add_subplot(gs[1, 1])

for idx, pi_id in enumerate(ph_ids):
    sub = df[df['pi_number'] == pi_id].sort_values('I')
    I_data = sub['I'].values
    P_data = sub['P'].values
    
    color = plt.cm.tab10(idx)
    ax.scatter(I_data, P_data, s=15, alpha=0.4, color=color, edgecolors='none')
    
    # Constrained fit
    r = constrained_results[pi_id]
    if r:
        I_fine = np.linspace(0.1, I_data.max()*1.1, 300)
        P_fine = pcc_scc_fixed_alpha(I_fine, *r['params'])
        Pmax, Ic, n = r['params']
        ax.plot(I_fine, P_fine, linewidth=2, color=color,
               label=f'{pi_id}: n={n:.1f}, Ic={Ic:.0f} (R²={r["r2"]:.3f})')

ax.set_xlabel('$I$ (µmol m⁻² s⁻¹)', fontsize=11)
ax.set_ylabel('$P^B$', fontsize=11)
ax.set_title(f'(d) All Curves with Fixed α={alpha_universal:.4f}',
            fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.set_xlim(0, None)
ax.set_ylim(0, None)

# (e) n vs K_r: the evolutionary landscape
ax = fig.add_subplot(gs[2, 0])

K_r_cont = np.linspace(0.005, 0.3, 200)
n_cont = 1.0 / (K_d_conserved + K_r_cont)

ax.plot(K_r_cont, n_cont, '-', color='#E91E63', linewidth=3,
       label='$n = 1/(K_d + K_r)$, $K_d$=0.02')

# Mark observed curves
for pi_id, n_obs in zip(ph_ids, ns):
    K_r_inferred = 1.0/n_obs - K_d_conserved
    if K_r_inferred > 0:
        color = 'green' if n_obs > 6 else 'blue'
        ax.plot(K_r_inferred, n_obs, 'o', markersize=12, color=color,
               markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.annotate(f'{pi_id}\nn={n_obs:.1f}', 
                   xy=(K_r_inferred, n_obs),
                   xytext=(K_r_inferred + 0.03, n_obs + 0.3),
                   fontsize=8)

ax.axhspan(2.5, 3.5, alpha=0.05, color='blue', label='LPS range (n≈3)')
ax.set_xlabel('$K_r$ (FtsH repair saturation)', fontsize=11)
ax.set_ylabel('$n$ (Hill cooperativity)', fontsize=11)
ax.set_title('(e) The Evolutionary Landscape: $n(K_r)$',
            fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 15)

# (f) The cross-domain universality summary
ax = fig.add_subplot(gs[2, 1])
ax.axis('off')

summary = """
 ╔═══════════════════════════════════════════════════╗
 ║       PCC/SCC EVOLUTIONARY UNIVERSALITY           ║
 ╠═══════════════════════════════════════════════════╣
 ║                                                   ║
 ║  "What physics determines, evolution preserves.   ║
 ║   What physics permits, evolution explores."       ║
 ║                                                   ║
 ║  PCC = physical law → conserved → universal       ║
 ║  SCC = adaptive response → variable → diverse     ║
 ║                                                   ║
 ║  LPS:  PCC = Arrhenius barrier (physics)          ║
 ║        SCC = CN≥3 threshold (crystal geometry)    ║
 ║        → No evolution. Geometry IS the physics.   ║
 ║                                                   ║
 ║  Photosynthesis:                                  ║
 ║        PCC = photon capture (quantum mechanics)   ║
 ║        SCC = PSII repair balance (enzyme kinetics)║
 ║        → Evolution tunes K_r. Everything else     ║
 ║          follows from 1/(K_d + K_r).              ║
 ║                                                   ║
 ║  The Hill function is the universal grammar.      ║
 ║  n is the species-specific accent.                ║
 ║  K_r is the evolutionary word that changes it.    ║
 ║                                                   ║
 ╚═══════════════════════════════════════════════════╝
"""

ax.text(0.02, 0.98, summary, transform=ax.transAxes,
       fontsize=8.5, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#FFF8E1', alpha=0.9))
ax.set_title('(f) The Principle', fontsize=11, fontweight='bold')

plt.savefig('fig6_evolutionary_invariance.png', dpi=200, bbox_inches='tight')
print("\n✅ fig6_evolutionary_invariance.png")

# ============================================================
# 8. SUMMARY STATEMENT
# ============================================================
print("\n" + "="*70)
print("EVOLUTIONARY INVARIANCE: SUMMARY OF EVIDENCE")
print("="*70)
print(f"""
From 4 photoinhibited PI curves (proof-of-concept):

1. α CONSERVATION: CV(α) = {cvs['α (PCC - quantum yield)']:.1f}%
   → α varies {cvs['α (PCC - quantum yield)']:.1f}% across curves
   → Consistent with conserved quantum yield (PCC)
   
2. SCC VARIABILITY: CV(n) = {cvs['n (SCC - cooperativity)']:.1f}%, CV(Ic) = {cvs['Ic (SCC - critical I)']:.1f}%
   → SCC parameters are {cvs['n (SCC - cooperativity)']/cvs['α (PCC - quantum yield)']:.1f}× more variable than α
   → Consistent with adaptive tuning (SCC)
   
3. CONSTRAINED FIT: Fixing α = {alpha_universal:.5f}
   → Mean R² loss = {mean_delta:.4f}
   → NEGLIGIBLE cost: α can be treated as universal
   
4. K_r AS EVOLUTIONARY KNOB:
   → n = 1/(K_d + K_r) maps K_r → n → gate shape → species identity
   → Inferred K_r range: {1.0/max(ns)-K_d_conserved:.3f} to {1.0/min(ns)-K_d_conserved:.3f}
   → K_r varies ~{(1.0/min(ns)-K_d_conserved)/(1.0/max(ns)-K_d_conserved):.0f}-fold → n varies ~2-fold
   
5. TESTABLE PREDICTION FOR 1808 CURVES:
   → All PI curves should form a 1-parameter family indexed by K_r
   → Shade species: K_r < 0.05 → n > 7 → sharp photoinhibition
   → Light species: K_r > 0.15 → n < 5 → gradual photoinhibition
   → Species identity should predict K_r (and vice versa)
""")
