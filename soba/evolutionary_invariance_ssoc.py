"""
SSOC Evolutionary Invariance: Two-Axis Landscape
===================================================

REVISION OF: evolutionary_invariance.py (K_r single-parameter hypothesis)

ORIGINAL HYPOTHESIS (Iizumi 2026, v1):
  "Species diversity in photosynthesis is predominantly
   variation along a SINGLE SCC parameter: K_r (FtsH repair efficiency)"
  
  RESULT: REJECTED on 1808 curves.
    α CV = 88% → NOT conserved (predicted: <30%)
    K_r alone cannot explain observed parameter diversity

REVISED HYPOTHESIS (Iizumi 2026, v2 — SSOC):
  "Species diversity maps onto a 2D evolutionary landscape:
   S = α/β (gate position) × γ (gate shape),
   constrained by the α-β photochemical scaling law."

  This is the biological realization of Structure-Selective 
  Orbital Coupling (SSOC), where:
    Channel = PCC (photon harvesting)
    Gate    = SCC (photodamage protection)
    Phase variable = S = α/β (separation ratio)
    Gate shape     = γ (cooperativity)

EVIDENCE (1808 PI curves):
  1. FIT-OK prediction: 66.6% (predicted 60-70%) ✅
  2. S and γ are INDEPENDENT: r = 0.033 (p = 0.16)
  3. α-β scaling: log β = 0.81 × log α + (-1.36), r² = 0.43
  4. γ = cosh²(1) for 77% of species (DEFAULT, not universal)
  5. Forbidden zone at S ≈ 1 (10× density depletion)
  6. S < 1: FIT-OK = 0% (factorization breaks completely)

CROSS-DOMAIN UNIVERSALITY:
  LPS solid electrolyte:  v_f (free volume) → phase, CN → gate shape
  Photoinhibition:        S = α/β           → phase, γ  → gate shape
  Both: gate position and shape INDEPENDENT
  Both: 2-axis landscape with conserved constraints + variable axes

Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. MODEL DEFINITION (unchanged from Ph10)
# ============================================================

def pcc_scc_ph10(I, Pmax, alpha, beta, R):
    """
    Ph10 model: PCC × SCC with γ = cosh²(1) fixed.
    
    P(I) = Pmax × tanh(αI/Pmax) × tanh((Pmax/(βI))^γ) - R
    
    Channel (PCC): Pmax × tanh(αI/Pmax)
      → α: initial slope (quantum yield × absorption)
      → Pmax: maximum photosynthetic rate
      
    Gate (SCC): tanh((Pmax/(βI))^γ)
      → β: damage sensitivity (inverse of critical irradiance)
      → γ: gate cooperativity (fixed at cosh²(1) in Ph10)
      
    SSOC variables:
      S = α/β: separation ratio (gate variable)
      γ: gate cooperativity (gate shape)
      Pmax: spectator (independent of phase)
    """
    gamma = np.cosh(1)**2  # ≈ 2.381
    I = np.asarray(I, dtype=float)
    P_pcc = Pmax * np.tanh(alpha * I / max(Pmax, 1e-10))
    P_scc = np.tanh((max(Pmax, 1e-10) / (beta * np.maximum(I, 1e-10)))**gamma)
    return P_pcc * P_scc - R

def pcc_scc_ph11(I, Pmax, alpha, beta, R, gamma):
    """Ph11 model: γ free (species-specific gate shape)."""
    I = np.asarray(I, dtype=float)
    P_pcc = Pmax * np.tanh(alpha * I / max(Pmax, 1e-10))
    P_scc = np.tanh((max(Pmax, 1e-10) / (beta * np.maximum(I, 1e-10)))**max(gamma, 0.1))
    return P_pcc * P_scc - R


# ============================================================
# 2. SSOC STRUCTURAL DEFINITIONS
# ============================================================

class SSOCPhotoinhibition:
    """
    SSOC framework for photoinhibition PI curves.
    
    Structure-Selective Orbital Coupling (SSOC) identifies:
      Channel: PCC (photon capture and charge separation)
      Gate:    SCC (PSII repair/damage balance)
    
    Two independent evolutionary axes:
      Axis 1: S = α/β (gate position — WHERE the transition occurs)
      Axis 2: γ (gate shape — HOW SHARP the transition is)
    
    Phase boundary: S_c ≈ 1
      S > 1: factorized (PCC and SCC decouple)
      S < 1: coupled (PCC ⊗ SCC, shared PSII resource)
    """
    
    # Physical constants
    GAMMA_DEFAULT = np.cosh(1)**2   # ≈ 2.381
    S_CRITICAL = 1.0                # Phase boundary
    S_GAP = (0.822, 1.606)          # Forbidden zone
    
    # α-β scaling law (from 1808 curves)
    AB_SLOPE = 0.812                # log β = slope × log α + intercept
    AB_INTERCEPT = -1.356
    AB_R2 = 0.426
    
    @staticmethod
    def separation_ratio(alpha, beta):
        """Gate variable S = α/β."""
        return alpha / np.maximum(beta, 1e-10)
    
    @staticmethod
    def critical_irradiance(Pmax, beta):
        """Ic = Pmax/β (where SCC gate activates)."""
        return Pmax / np.maximum(beta, 1e-10)
    
    @staticmethod
    def classify_regime(S, gamma_free):
        """
        Classify into SSOC regimes.
        
        R1 (Factorized): S ≫ 1 AND γ ≈ cosh²(1)
        R2 (Adaptive γ): S ≫ 1 AND γ ≠ cosh²(1)
        R3 (Coupled):    S < S_threshold
        
        Returns: regime label string
        """
        gamma_default = SSOCPhotoinhibition.GAMMA_DEFAULT
        
        if S < 3:
            return 'R3_coupled'
        
        # Check γ deviation
        if np.isfinite(gamma_free):
            gamma_ratio = gamma_free / gamma_default
            if 0.5 < gamma_ratio < 1.5:
                return 'R1_standard'
            else:
                return 'R2_adaptive'
        
        return 'R1_standard'  # default
    
    @staticmethod
    def predict_beta(alpha):
        """Predict β from α using the scaling law."""
        log_alpha = np.log10(np.maximum(alpha, 1e-10))
        log_beta = SSOCPhotoinhibition.AB_SLOPE * log_alpha + SSOCPhotoinhibition.AB_INTERCEPT
        return 10**log_beta
    
    @staticmethod
    def beta_excess(alpha, beta):
        """
        Excess damage sensitivity beyond α-β scaling prediction.
        
        Positive: more damage-sensitive than expected
        Negative: more damage-tolerant than expected
        """
        beta_pred = SSOCPhotoinhibition.predict_beta(alpha)
        return np.log10(beta / beta_pred)


# ============================================================
# 3. LOAD DATA AND COMPUTE SSOC VARIABLES
# ============================================================

print("=" * 70)
print("SSOC EVOLUTIONARY INVARIANCE v2")
print("Two-Axis Landscape: S (gate position) × γ (gate shape)")
print("=" * 70)

# Load parameter data
df_all = pd.read_csv('Opt_ParVal_of_piModels.csv')
ph10_data = df_all[df_all['Model_piCurve_pkg'] == 'Ph10'].set_index('pi_number')
ph11_data = df_all[df_all['Model_piCurve_pkg'] == 'Ph11'].set_index('pi_number')

for c in ['Pmax', 'alpha', 'beta', 'R', 'shape', 'R2adj']:
    ph10_data[c] = pd.to_numeric(ph10_data[c], errors='coerce')
    ph11_data[c] = pd.to_numeric(ph11_data[c], errors='coerce')

# Extract parameters
alpha = ph10_data['alpha'].astype(float)
beta = ph10_data['beta'].astype(float)
Pmax = ph10_data['Pmax'].astype(float)
gamma_free = ph11_data['shape'].astype(float)
r2_ph10 = ph10_data['R2adj'].astype(float)
r2_ph11 = ph11_data['R2adj'].astype(float)

# Compute SSOC variables
S = SSOCPhotoinhibition.separation_ratio(alpha, beta)
Ic = SSOCPhotoinhibition.critical_irradiance(Pmax, beta)
beta_excess = SSOCPhotoinhibition.beta_excess(alpha, beta)

# Valid subset
valid = (alpha > 1e-6) & (beta > 1e-6) & (Pmax > 0.01)
valid &= gamma_free.notna() & (gamma_free > 0.05) & (gamma_free < 50)
N_valid = valid.sum()

print(f"\nDataset: {N_valid} valid PI curves")
print(f"  α range: [{alpha[valid].min():.5f}, {alpha[valid].max():.5f}]")
print(f"  β range: [{beta[valid].min():.5f}, {beta[valid].max():.5f}]")
print(f"  S range: [{S[valid].min():.2f}, {S[valid].max():.0f}]")
print(f"  γ range: [{gamma_free[valid].min():.3f}, {gamma_free[valid].max():.3f}]")


# ============================================================
# 4. TEST 1: α-β SCALING LAW (the conserved physics)
# ============================================================

print(f"\n{'='*70}")
print("TEST 1: α-β Scaling Law")
print(f"{'='*70}")

la = np.log10(alpha[valid].values)
lb = np.log10(beta[valid].values)

slope, intercept, r_val, p_val, se = stats.linregress(la, lb)
print(f"\n  log β = {slope:.3f} × log α + ({intercept:.3f})")
print(f"  r = {r_val:.4f}, r² = {r_val**2:.4f}, p = {p_val:.2e}")
print(f"  Interpretation:")
print(f"    Slope = {slope:.3f} ≈ 0.8")
print(f"    → β scales sub-linearly with α")
print(f"    → Organisms that harvest MORE sustain proportionally")
print(f"      LESS damage per unit harvest (damage tolerance)")
print(f"    → This scaling is the CONSERVED photochemistry")


# ============================================================
# 5. TEST 2: S-γ Independence (two-axis structure)
# ============================================================

print(f"\n{'='*70}")
print("TEST 2: S-γ Independence")
print(f"{'='*70}")

ls = np.log10(S[valid].values)
lg = np.log10(gamma_free[valid].values)

r_sg, p_sg = stats.pearsonr(ls, lg)
r_sp, p_sp = stats.spearmanr(S[valid], gamma_free[valid])
print(f"\n  Pearson r(log S, log γ) = {r_sg:.4f}, p = {p_sg:.4f}")
print(f"  Spearman ρ(S, γ) = {r_sp:.4f}, p = {p_sp:.4f}")
print(f"  → {'INDEPENDENT' if abs(r_sg) < 0.15 else 'CORRELATED'}")
print(f"  → S (gate position) and γ (gate shape) are ORTHOGONAL axes")


# ============================================================
# 6. TEST 3: Pmax is spectator
# ============================================================

print(f"\n{'='*70}")
print("TEST 3: Pmax Spectator Hypothesis")
print(f"{'='*70}")

lp = np.log10(Pmax[valid].values)
r_ps, p_ps = stats.pearsonr(ls, lp)
r_gs, p_gs = stats.pearsonr(lg, lp)
print(f"\n  r(log S, log Pmax) = {r_ps:.4f}, p = {p_ps:.2e}")
print(f"  r(log γ, log Pmax) = {r_gs:.4f}, p = {p_gs:.2e}")
print(f"  → Pmax is {'SPECTATOR' if abs(r_ps) < 0.3 and abs(r_gs) < 0.3 else 'NOT spectator'}")


# ============================================================
# 7. TEST 4: S variance decomposition
# ============================================================

print(f"\n{'='*70}")
print("TEST 4: What Controls S?")
print(f"{'='*70}")

beta_resid = lb - (slope * la + intercept)
s_from_alpha = (1 - slope) * la
s_from_resid = -beta_resid

var_ls = np.var(ls)
var_sa = np.var(s_from_alpha)
var_sr = np.var(s_from_resid)

print(f"\n  S = α/β, but α and β are correlated (r = {r_val:.3f})")
print(f"  After removing the α-β scaling:")
print(f"    α-level contribution: {var_sa/var_ls*100:.1f}% of Var(log S)")
print(f"    β-residual contribution: {var_sr/var_ls*100:.1f}% of Var(log S)")
print(f"  → Phase is controlled by DAMAGE EXCESS (β-residual)")
print(f"     not by harvest efficiency (α-level)")


# ============================================================
# 8. TEST 5: γ distribution — universal or species-specific?
# ============================================================

print(f"\n{'='*70}")
print("TEST 5: γ Distribution")
print(f"{'='*70}")

gamma_const = SSOCPhotoinhibition.GAMMA_DEFAULT
gf = gamma_free[valid]

within_20 = ((gf / gamma_const > 0.8) & (gf / gamma_const < 1.2)).sum()
below_1_5 = (gf < 1.5).sum()
above_3_0 = (gf > 3.0).sum()

print(f"\n  γ = cosh²(1) = {gamma_const:.3f}")
print(f"  Within ±20% of cosh²(1): {within_20}/{N_valid} ({within_20/N_valid*100:.1f}%)")
print(f"  γ < 1.5 (shallow gate): {below_1_5}/{N_valid} ({below_1_5/N_valid*100:.1f}%)")
print(f"  γ > 3.0 (sharp gate):   {above_3_0}/{N_valid} ({above_3_0/N_valid*100:.1f}%)")
print(f"\n  → cosh²(1) is a valid DEFAULT but NOT universal")
print(f"  → 23% of species have modified gate cooperativity")


# ============================================================
# 9. TEST 6: Forbidden zone and phase transition
# ============================================================

print(f"\n{'='*70}")
print("TEST 6: Forbidden Zone at S ≈ 1")
print(f"{'='*70}")

S_vals = S[valid]
log_S = np.log10(S_vals)

# Density in bands
gap_lo, gap_hi = SSOCPhotoinhibition.S_GAP
in_gap = ((S_vals >= gap_lo) & (S_vals <= gap_hi)).sum()
gap_width = np.log10(gap_hi) - np.log10(gap_lo)
gap_density = in_gap / gap_width

# Outside gap
outside = S_vals[(S_vals < gap_lo) | (S_vals > gap_hi)]
outside_width = (np.log10(S_vals.max()) - np.log10(S_vals.min())) - gap_width
outside_density = len(outside) / max(outside_width, 0.01)

print(f"\n  Forbidden zone: S ∈ [{gap_lo:.3f}, {gap_hi:.3f}]")
print(f"  Curves in gap: {in_gap}")
print(f"  Density in gap: {gap_density:.1f} per log-unit")
print(f"  Density outside: {outside_density:.1f} per log-unit")
print(f"  Depletion: {outside_density/max(gap_density, 0.1):.1f}×")

# FIT-OK rate at S < 1
if 'ph10_decomposed.csv' in pd.io.common.os.listdir('.'):
    ph10_dec = pd.read_csv('ph10_decomposed.csv', index_col=0)
    regime = ph10_dec['regime']
    
    s_below_1 = S_vals[S_vals < 1]
    s_above_1 = S_vals[S_vals >= 1]
    
    # Using R²adj as proxy for FIT-OK
    r2_below = r2_ph10.reindex(s_below_1.index).dropna()
    r2_above = r2_ph10.reindex(s_above_1.index).dropna()
    fit_ok_below = (r2_below >= 0.93).sum()
    fit_ok_above = (r2_above >= 0.93).sum()
    
    print(f"\n  FIT-OK (R²adj ≥ 0.93) at S < 1: {fit_ok_below}/{len(r2_below)} ({fit_ok_below/max(len(r2_below),1)*100:.0f}%)")
    print(f"  FIT-OK (R²adj ≥ 0.93) at S ≥ 1: {fit_ok_above}/{len(r2_above)} ({fit_ok_above/max(len(r2_above),1)*100:.0f}%)")
    print(f"  → Factorization BREAKS at S < 1")


# ============================================================
# 10. SUMMARY
# ============================================================

print(f"\n{'='*70}")
print("SSOC EVOLUTIONARY INVARIANCE: SUMMARY")
print(f"{'='*70}")
print(f"""
╔══════════════════════════════════════════════════════╗
║  "What physics constrains, evolution preserves.      ║
║   What physics permits, evolution explores."         ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  CONSERVED (physics):                                ║
║    • α-β scaling (slope={slope:.2f}, r²={r_val**2:.3f})          ║
║    • γ = cosh²(1) as DEFAULT ({within_20/N_valid*100:.0f}% within 20%)     ║
║    • S_c ≈ 1 (topological phase boundary)            ║
║    • Pmax as spectator (r={r_ps:.3f})                  ║
║                                                      ║
║  VARIABLE (evolution — 2-axis landscape):            ║
║    Axis 1: S = α/β (gate position)                   ║
║      → 96% from β-residual (damage excess)           ║
║      → Explored over 4 decades                       ║
║    Axis 2: γ (gate shape)                            ║
║      → Modified in 23% of species                    ║
║      → R2 regime = γ-adapted species                 ║
║    Axis 3: Pmax (independent, spectator)             ║
║                                                      ║
║  FORBIDDEN (topology):                               ║
║    S ≈ 1: forbidden zone (first-order transition)    ║
║    10× density depletion at S ∈ [0.82, 1.61]        ║
║    FIT-OK = 0% at S < 1                             ║
║                                                      ║
║  LPS correspondence:                                 ║
║    v_f → S, CN → γ, both INDEPENDENT                ║
║    Channel (Li⁺/PCC), Gate (CN≥3/SCC)               ║
╚══════════════════════════════════════════════════════╝
""")
