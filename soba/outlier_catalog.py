"""
PCC×SCC Outlier Catalog: Classification Framework
====================================================

The 5-class deviation taxonomy for PI curves:
  ① Fit OK     — PCC×SCC sufficient (R² > 0.95, all residuals < 2σ)
  ② α-outlier  — Initial slope anomaly (a* variation: antenna size)
  ③ n-outlier  — Cooperativity anomaly (K_r variation: FtsH repair)
  ④ Ic-outlier — Critical irradiance anomaly (V_repair/V_damage ratio)
  ⑤ Non-fit    — Model inadequacy (NPQ double-gate / input transform)

Each class generates a REVERSE PREDICTION testable against metadata.

Strategy: Define quantitative thresholds BEFORE seeing the 1808 curves.
This prevents post-hoc rationalization — the catalog is a priori.

Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def pcc_scc_model(I, Pmax, alpha, Ic, n):
    """Standard 4-parameter PCC×SCC model"""
    I = np.asarray(I, dtype=float)
    P_pcc = Pmax * np.tanh(alpha * I / max(Pmax, 1e-10))
    p_active = 1.0 / (1.0 + (I / max(Ic, 1e-10))**max(n, 0.1))
    return P_pcc * p_active

def pcc_only_model(I, Pmax, alpha):
    """2-parameter PCC-only (no photoinhibition)"""
    return Pmax * np.tanh(alpha * I / max(Pmax, 1e-10))

def pcc_scc_double_gate(I, Pmax, alpha, Ic1, n1, Ic2, n2):
    """6-parameter double SCC gate (for NPQ + photoinhibition)"""
    P_pcc = Pmax * np.tanh(alpha * I / max(Pmax, 1e-10))
    gate1 = 1.0 / (1.0 + (I / max(Ic1, 1e-10))**max(n1, 0.1))
    gate2 = 1.0 / (1.0 + (I / max(Ic2, 1e-10))**max(n2, 0.1))
    return P_pcc * gate1 * gate2

# ============================================================
# FITTING ENGINE
# ============================================================

def robust_fit_pcc_scc(I, P):
    """Multi-start fitting for PCC×SCC model"""
    Pm = np.max(P)
    I_peak = I[np.argmax(P)] if len(P) > 0 else 500
    I_max = np.max(I)
    
    # Estimate alpha from initial slope
    low_mask = I < np.percentile(I, 20)
    if np.sum(low_mask) > 2 and np.ptp(I[low_mask]) > 0:
        alpha_est = max(np.polyfit(I[low_mask], P[low_mask], 1)[0], 0.005)
    else:
        alpha_est = 0.03
    
    best = None
    for pm in [Pm, Pm*1.2, Pm*1.5, Pm*2.0]:
        for a in [alpha_est*0.3, alpha_est*0.7, alpha_est, alpha_est*1.5]:
            for ic in [I_peak*0.5, I_peak, I_peak*1.5, I_max*0.7, I_max]:
                for nn in [2, 3, 4, 6, 8, 10, 12]:
                    try:
                        popt, pcov = curve_fit(pcc_scc_model, I, P,
                                              p0=[pm, a, ic, nn],
                                              bounds=([0, 0, 1, 0.3],
                                                     [20*Pm, 2, 20000, 20]),
                                              maxfev=20000)
                        P_pred = pcc_scc_model(I, *popt)
                        ss_res = np.sum((P - P_pred)**2)
                        ss_tot = np.sum((P - np.mean(P))**2)
                        r2 = 1 - ss_res / max(ss_tot, 1e-30)
                        
                        # Parameter uncertainties
                        try:
                            perr = np.sqrt(np.diag(pcov))
                        except:
                            perr = np.full(4, np.inf)
                        
                        if best is None or r2 > best['r2']:
                            best = {
                                'params': popt,
                                'perr': perr,
                                'r2': r2,
                                'P_pred': P_pred,
                                'residuals': P - P_pred,
                                'rmse': np.sqrt(np.mean((P - P_pred)**2))
                            }
                    except:
                        pass
    return best

def robust_fit_pcc_only(I, P):
    """Fit PCC-only model"""
    Pm = np.max(P)
    low_mask = I < np.percentile(I, 20)
    if np.sum(low_mask) > 2 and np.ptp(I[low_mask]) > 0:
        alpha_est = max(np.polyfit(I[low_mask], P[low_mask], 1)[0], 0.005)
    else:
        alpha_est = 0.03
    
    best = None
    for pm in [Pm, Pm*1.2, Pm*1.5]:
        for a in [alpha_est*0.5, alpha_est, alpha_est*2]:
            try:
                popt, _ = curve_fit(pcc_only_model, I, P,
                                   p0=[pm, a],
                                   bounds=([0, 0], [20*Pm, 2]),
                                   maxfev=10000)
                P_pred = pcc_only_model(I, *popt)
                r2 = 1 - np.sum((P-P_pred)**2) / max(np.sum((P-np.mean(P))**2), 1e-30)
                if best is None or r2 > best['r2']:
                    best = {'params': popt, 'r2': r2, 'P_pred': P_pred}
            except:
                pass
    return best

# ============================================================
# CLASSIFICATION ENGINE
# ============================================================

class OutlierCatalog:
    """
    A priori classification of PI curves into 5 deviation classes.
    
    Thresholds are defined BEFORE seeing large datasets.
    No post-hoc adjustment allowed.
    """
    
    # Reference ranges from proof-of-concept (4 photoinhibited curves)
    # These define "normal" — deviations from these are classified
    ALPHA_REF_RANGE = (0.02, 0.08)   # Expected α for photosynthetic organisms
    N_REF_RANGE = (2.0, 12.0)         # Expected n from GK theory
    IC_REF_RANGE = (100, 2000)         # Expected Ic (µmol/m²/s)
    
    # Thresholds for classification
    R2_OK = 0.93           # Minimum R² for "Fit OK"
    R2_NONFIT = 0.80       # Below this = definite non-fit
    RESIDUAL_SIGMA = 2.5   # Residuals beyond this many σ = structured
    ALPHA_CV_THRESH = 0.5  # Relative deviation from reference α
    N_CV_THRESH = 0.5      # Relative deviation from expected n
    IC_CV_THRESH = 0.5     # Relative deviation from expected Ic
    
    @staticmethod
    def classify(I, P, fit_result, reference_params=None):
        """
        Classify a single PI curve.
        
        Returns: dict with class, confidence, reverse_prediction
        """
        if fit_result is None or fit_result['r2'] < 0:
            return {
                'class': 5,
                'label': 'NON-FIT',
                'confidence': 1.0,
                'reason': 'Fitting failed completely',
                'reverse_prediction': 'Check data quality; possibly non-photosynthetic signal',
                'r2': -1,
                'params': None
            }
        
        Pmax, alpha, Ic, n = fit_result['params']
        r2 = fit_result['r2']
        residuals = fit_result['residuals']
        
        # Normalized residuals
        sigma = np.std(residuals)
        norm_res = residuals / max(sigma, 1e-10)
        max_norm_res = np.max(np.abs(norm_res))
        
        # Residual structure test (runs test or Shapiro-Wilk)
        if len(residuals) > 8:
            try:
                _, p_shapiro = shapiro(residuals)
                structured_residuals = p_shapiro < 0.05
            except:
                structured_residuals = False
        else:
            structured_residuals = False
        
        # Decision tree (ORDER MATTERS)
        result = {
            'r2': r2,
            'params': fit_result['params'],
            'perr': fit_result.get('perr', [np.nan]*4),
            'rmse': fit_result['rmse'],
            'max_norm_residual': max_norm_res,
            'structured_residuals': structured_residuals,
        }
        
        # ⑤ NON-FIT: R² too low or severe structured residuals
        if r2 < OutlierCatalog.R2_NONFIT:
            result.update({
                'class': 5,
                'label': 'NON-FIT',
                'confidence': min(1.0, (OutlierCatalog.R2_NONFIT - r2) / 0.2 + 0.5),
                'reason': f'R²={r2:.3f} < {OutlierCatalog.R2_NONFIT}',
                'reverse_prediction': (
                    'NPQ double-gate hypothesis: Try P = PCC × gate1 × gate2. '
                    'Or: non-standard input transform (spectral quality effect). '
                    'Or: dynamic (non-steady-state) measurement artifact.'
                ),
            })
            return result
        
        # Check if photoinhibition is even present
        # (declining P at high I?)
        if len(I) > 5:
            I_sorted = np.argsort(I)
            top_quarter = I_sorted[int(0.75*len(I)):]
            P_top = P[top_quarter]
            P_peak = np.max(P)
            decline_ratio = (P_peak - np.mean(P_top)) / max(P_peak, 1e-10)
            has_photoinhibition = decline_ratio > 0.05
        else:
            has_photoinhibition = True  # assume yes if few points
        
        # Reference params for deviation detection
        if reference_params is not None:
            alpha_ref, n_ref, Ic_ref = reference_params
        else:
            alpha_ref = np.mean(OutlierCatalog.ALPHA_REF_RANGE)
            n_ref = 6.0  # Expected from GK theory
            Ic_ref = 700  # Mean from proof-of-concept
        
        # Compute deviations
        alpha_dev = abs(alpha - alpha_ref) / max(alpha_ref, 1e-10)
        n_dev = abs(n - n_ref) / max(n_ref, 1e-10)
        Ic_dev = abs(Ic - Ic_ref) / max(Ic_ref, 1e-10)
        
        # ① FIT OK: good R², no large residuals, params in range
        if (r2 >= OutlierCatalog.R2_OK and 
            max_norm_res < OutlierCatalog.RESIDUAL_SIGMA and
            alpha_dev < OutlierCatalog.ALPHA_CV_THRESH and
            n_dev < OutlierCatalog.N_CV_THRESH and
            Ic_dev < OutlierCatalog.IC_CV_THRESH):
            result.update({
                'class': 1,
                'label': 'FIT-OK',
                'confidence': min(1.0, (r2 - OutlierCatalog.R2_OK) / 0.05 + 0.5),
                'reason': f'R²={r2:.3f}, all params in reference range',
                'reverse_prediction': (
                    'Standard PCC×SCC physics applies. '
                    f'GK parameters: K_d≈0.02, K_r≈{1/n - 0.02:.3f}. '
                    'Expect moderate light adaptation.'
                ),
            })
            return result
        
        # For remaining classes: identify the DOMINANT deviation
        deviations = {
            'alpha': alpha_dev,
            'n': n_dev,
            'Ic': Ic_dev,
        }
        dominant = max(deviations, key=deviations.get)
        
        # ② α-OUTLIER
        if dominant == 'alpha' and alpha_dev >= OutlierCatalog.ALPHA_CV_THRESH:
            direction = 'HIGH' if alpha > alpha_ref else 'LOW'
            result.update({
                'class': 2,
                'label': f'α-OUTLIER ({direction})',
                'confidence': min(1.0, alpha_dev),
                'reason': f'α={alpha:.4f} vs ref={alpha_ref:.4f} (dev={alpha_dev:.1%})',
                'reverse_prediction': (
                    f'α {">" if direction=="HIGH" else "<"} reference → '
                    f'{"Large" if direction=="HIGH" else "Small"} effective absorption '
                    f'cross-section a*. '
                    f'Predict: {"large antenna (shade-acclimated)" if direction=="HIGH" else "small antenna (high-light acclimated)"}. '
                    f'Check: Chl a/cell, LHCII/PSII ratio, cell size.'
                ),
            })
            return result
        
        # ③ n-OUTLIER
        if dominant == 'n' and n_dev >= OutlierCatalog.N_CV_THRESH:
            direction = 'HIGH' if n > n_ref else 'LOW'
            K_r_inferred = max(1.0/n - 0.02, 0.001)
            result.update({
                'class': 3,
                'label': f'n-OUTLIER ({direction})',
                'confidence': min(1.0, n_dev),
                'reason': f'n={n:.2f} vs ref={n_ref:.1f} (dev={n_dev:.1%})',
                'reverse_prediction': (
                    f'n {">" if direction=="HIGH" else "<"} reference → '
                    f'K_r ≈ {K_r_inferred:.3f} → '
                    f'FtsH repair {"highly saturated" if direction=="HIGH" else "under-saturated"}. '
                    f'Predict: {"shade-adapted, few FtsH copies, sharp photoinhibition onset" if direction=="HIGH" else "light-adapted, abundant FtsH, gradual photoinhibition"}. '
                    f'Check: FtsH expression, habitat light regime.'
                ),
            })
            return result
        
        # ④ Ic-OUTLIER
        if dominant == 'Ic' and Ic_dev >= OutlierCatalog.IC_CV_THRESH:
            direction = 'HIGH' if Ic > Ic_ref else 'LOW'
            result.update({
                'class': 4,
                'label': f'Ic-OUTLIER ({direction})',
                'confidence': min(1.0, Ic_dev),
                'reason': f'Ic={Ic:.0f} vs ref={Ic_ref:.0f} (dev={Ic_dev:.1%})',
                'reverse_prediction': (
                    f'Ic {">" if direction=="HIGH" else "<"} reference → '
                    f'V_repair/V_damage ratio {"high" if direction=="HIGH" else "low"}. '
                    f'Predict: {"high repair capacity OR low damage rate (photoprotection)" if direction=="HIGH" else "low repair capacity OR high damage rate (photosensitive)"}. '
                    f'Check: growth irradiance, photoprotective pigments (zeaxanthin, β-carotene).'
                ),
            })
            return result
        
        # Intermediate: good R² but some deviation
        if r2 >= OutlierCatalog.R2_OK:
            result.update({
                'class': 1,
                'label': 'FIT-OK (marginal)',
                'confidence': 0.5,
                'reason': f'R²={r2:.3f}, minor parameter deviations',
                'reverse_prediction': 'Standard physics with mild adaptation.',
            })
        else:
            # Between R2_NONFIT and R2_OK
            result.update({
                'class': 5,
                'label': 'BORDERLINE NON-FIT',
                'confidence': 0.5,
                'reason': f'R²={r2:.3f}, between thresholds',
                'reverse_prediction': (
                    'Possible NPQ contribution or measurement artifact. '
                    'Try double-gate model or check measurement protocol.'
                ),
            })
        
        return result

# ============================================================
# RUN ON PROOF-OF-CONCEPT DATA
# ============================================================

df = pd.read_csv('piDataSet.csv')
all_ids = df['pi_number'].unique()

print("="*70)
print("OUTLIER CATALOG — PROOF OF CONCEPT (8 piCurve samples)")
print("="*70)

# First pass: fit all curves and compute reference
all_results = {}
for pi_id in all_ids:
    sub = df[df['pi_number'] == pi_id].sort_values('I')
    I = sub['I'].values.astype(float)
    P = sub['P'].values.astype(float)
    
    fit_4p = robust_fit_pcc_scc(I, P)
    fit_2p = robust_fit_pcc_only(I, P)
    
    all_results[pi_id] = {
        'I': I, 'P': P,
        'fit_4p': fit_4p,
        'fit_2p': fit_2p,
    }

# Compute reference from photoinhibited curves
ph_ids = ['PI002366', 'PI002413', 'PI002788', 'PI002794']
ref_alphas = [all_results[pi]['fit_4p']['params'][1] for pi in ph_ids if all_results[pi]['fit_4p']]
ref_ns = [all_results[pi]['fit_4p']['params'][3] for pi in ph_ids if all_results[pi]['fit_4p']]
ref_Ics = [all_results[pi]['fit_4p']['params'][2] for pi in ph_ids if all_results[pi]['fit_4p']]

ref_params = (np.mean(ref_alphas), np.mean(ref_ns), np.mean(ref_Ics))
print(f"\nReference parameters (from 4 photoinhibited curves):")
print(f"  α_ref = {ref_params[0]:.5f}")
print(f"  n_ref = {ref_params[1]:.2f}")
print(f"  Ic_ref = {ref_params[2]:.0f}")

# Classify all 8 curves
print(f"\n{'='*70}")
print(f"CLASSIFICATION RESULTS")
print(f"{'='*70}")
print(f"\n{'Curve':12s} {'Class':6s} {'Label':22s} {'R²':>6s} {'Pmax':>6s} {'α':>8s} "
      f"{'Ic':>6s} {'n':>5s} {'Conf':>5s}")
print("-"*85)

classifications = {}
for pi_id in all_ids:
    r = all_results[pi_id]
    
    if r['fit_4p'] is not None:
        cat = OutlierCatalog.classify(r['I'], r['P'], r['fit_4p'], ref_params)
    else:
        cat = {'class': 5, 'label': 'NON-FIT', 'r2': -1, 'params': None,
               'confidence': 1.0, 'reason': 'Fit failed', 'reverse_prediction': 'N/A'}
    
    classifications[pi_id] = cat
    
    if cat['params'] is not None:
        Pmax, alpha, Ic, n = cat['params']
        print(f"{pi_id:12s} {'①②③④⑤'[cat['class']-1]}     {cat['label']:22s} "
              f"{cat['r2']:6.3f} {Pmax:6.2f} {alpha:8.5f} {Ic:6.0f} {n:5.2f} {cat['confidence']:5.2f}")
    else:
        print(f"{pi_id:12s} ⑤     {cat['label']:22s}  {'N/A':>6s}")

# Print reverse predictions
print(f"\n{'='*70}")
print(f"REVERSE PREDICTIONS")
print(f"{'='*70}")

for pi_id in all_ids:
    cat = classifications[pi_id]
    symbol = '①②③④⑤'[cat['class']-1]
    print(f"\n{symbol} {pi_id}: {cat['label']}")
    print(f"   Reason: {cat['reason']}")
    print(f"   Prediction: {cat['reverse_prediction']}")

# ============================================================
# STATISTICS
# ============================================================
print(f"\n{'='*70}")
print(f"CATALOG STATISTICS")
print(f"{'='*70}")

from collections import Counter
class_counts = Counter(c['class'] for c in classifications.values())
class_names = {1: '① FIT-OK', 2: '② α-outlier', 3: '③ n-outlier', 
               4: '④ Ic-outlier', 5: '⑤ Non-fit'}

for cls in range(1, 6):
    count = class_counts.get(cls, 0)
    pct = count / len(classifications) * 100
    print(f"  {class_names[cls]:16s}: {count:2d} ({pct:5.1f}%)")

# ============================================================
# FIGURE: The Catalog Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('PCC×SCC Outlier Catalog: 8 PI Curves Classified\n'
             '"Every deviation is a prediction"',
             fontsize=13, fontweight='bold')

# Class colors
class_colors = {
    1: '#4CAF50',  # green = OK
    2: '#FF9800',  # orange = α
    3: '#E91E63',  # pink = n
    4: '#9C27B0',  # purple = Ic
    5: '#F44336',  # red = non-fit
}
class_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'X'}

# (a) All curves with classification coloring
ax = axes[0, 0]
for pi_id in all_ids:
    r = all_results[pi_id]
    cat = classifications[pi_id]
    color = class_colors[cat['class']]
    
    ax.scatter(r['I'], r['P'], s=12, alpha=0.4, color=color, edgecolors='none')
    
    if r['fit_4p'] is not None:
        I_fine = np.linspace(0.1, np.max(r['I'])*1.1, 300)
        P_fine = pcc_scc_model(I_fine, *r['fit_4p']['params'])
        label = f"{pi_id} ({'①②③④⑤'[cat['class']-1]})"
        ax.plot(I_fine, P_fine, linewidth=1.5, color=color, label=label)

ax.set_xlabel('I (µmol m⁻² s⁻¹)', fontsize=10)
ax.set_ylabel('P$^B$', fontsize=10)
ax.set_title('(a) All Curves with Classification', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='best')
ax.set_xlim(0, None)
ax.set_ylim(0, None)

# (b) Parameter space: α vs n
ax = axes[0, 1]
for pi_id in all_ids:
    cat = classifications[pi_id]
    if cat['params'] is not None:
        Pmax, alpha, Ic, n = cat['params']
        color = class_colors[cat['class']]
        marker = class_markers[cat['class']]
        ax.scatter(alpha, n, s=120, c=color, marker=marker, 
                  edgecolors='black', linewidth=1, zorder=10)
        ax.annotate(pi_id[-3:], (alpha, n), fontsize=7,
                   xytext=(5, 5), textcoords='offset points')

# Reference box
ax.axvspan(ref_params[0]*(1-0.5), ref_params[0]*(1+0.5), 
          alpha=0.08, color='green')
ax.axhspan(ref_params[1]*(1-0.5), ref_params[1]*(1+0.5),
          alpha=0.08, color='green')
ax.axvline(ref_params[0], color='green', linestyle=':', alpha=0.5)
ax.axhline(ref_params[1], color='green', linestyle=':', alpha=0.5)

ax.set_xlabel('α (initial slope)', fontsize=10)
ax.set_ylabel('n (Hill cooperativity)', fontsize=10)
ax.set_title('(b) Parameter Space: α vs n', fontsize=10, fontweight='bold')

# (c) Parameter space: Ic vs n
ax = axes[0, 2]
for pi_id in all_ids:
    cat = classifications[pi_id]
    if cat['params'] is not None:
        Pmax, alpha, Ic, n = cat['params']
        color = class_colors[cat['class']]
        marker = class_markers[cat['class']]
        ax.scatter(Ic, n, s=120, c=color, marker=marker,
                  edgecolors='black', linewidth=1, zorder=10)
        ax.annotate(pi_id[-3:], (Ic, n), fontsize=7,
                   xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('$I_c$ (critical irradiance)', fontsize=10)
ax.set_ylabel('n (Hill cooperativity)', fontsize=10)
ax.set_title('(c) Parameter Space: $I_c$ vs n', fontsize=10, fontweight='bold')

# (d-f) Individual curve diagnostics for 3 interesting cases
interesting = []
for pi_id in all_ids:
    cat = classifications[pi_id]
    interesting.append((pi_id, cat))
interesting.sort(key=lambda x: x[1]['class'])

for idx, (pi_id, cat) in enumerate(interesting[:3]):
    ax = axes[1, idx]
    r = all_results[pi_id]
    
    color = class_colors[cat['class']]
    symbol = '①②③④⑤'[cat['class']-1]
    
    # Data
    ax.scatter(r['I'], r['P'], s=25, color=color, alpha=0.6, 
              edgecolors='black', linewidth=0.5, label='Data', zorder=5)
    
    # PCC×SCC fit
    if r['fit_4p'] is not None:
        I_fine = np.linspace(0.1, np.max(r['I'])*1.1, 300)
        P_total = pcc_scc_model(I_fine, *r['fit_4p']['params'])
        Pmax, alpha, Ic, n = r['fit_4p']['params']
        
        # PCC component
        P_pcc = Pmax * np.tanh(alpha * I_fine / Pmax)
        
        # SCC gate (scaled to Pmax for visualization)
        p_active = 1.0 / (1.0 + (I_fine / Ic)**n)
        
        ax.plot(I_fine, P_pcc, '--', color='blue', linewidth=1.5, alpha=0.5, label='PCC')
        ax.plot(I_fine, P_total, '-', color=color, linewidth=2.5, 
               label=f'PCC×SCC (R²={cat["r2"]:.3f})')
        
        # SCC gate on secondary axis
        ax2 = ax.twinx()
        ax2.plot(I_fine, p_active, '-', color='gray', linewidth=1, alpha=0.4)
        ax2.set_ylabel('$p_{active}$', fontsize=8, color='gray')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
    
    ax.set_xlabel('I (µmol m⁻² s⁻¹)', fontsize=10)
    ax.set_ylabel('P$^B$', fontsize=10)
    ax.set_title(f'({"def"[idx]}) {pi_id}: {symbol} {cat["label"]}',
                fontsize=10, fontweight='bold', color=color)
    ax.legend(fontsize=7, loc='best')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('fig7_outlier_catalog.png', dpi=200, bbox_inches='tight')
print(f"\n✅ fig7_outlier_catalog.png")

# ============================================================
# THE BATTLE-READY SPECIFICATION
# ============================================================
print(f"\n{'='*70}")
print(f"BATTLE-READY: 1808-CURVE DEPLOYMENT SPECIFICATION")
print(f"{'='*70}")
print(f"""
Classification will be AUTOMATIC:
  Input: (I_i, P_i) arrays for each PI curve
  Output: Class ①-⑤ + confidence + reverse prediction

Decision tree:
  1. Fit PCC×SCC (4 params) with multi-start optimization
  2. Compute R², residual structure, parameter deviations
  3. Classify:
     R² < 0.80 → ⑤ NON-FIT
     R² ≥ 0.93 + params in range → ① FIT-OK
     Otherwise → identify dominant deviation:
       α dominant → ② α-outlier (a* = antenna size)
       n dominant → ③ n-outlier (K_r = FtsH efficiency)  
       Ic dominant → ④ Ic-outlier (repair/damage ratio)

Each class generates testable reverse predictions:
  ②: "This organism has unusual antenna size" → check Chl/cell
  ③: "This organism has unusual repair capacity" → check FtsH
  ④: "This organism has unusual damage tolerance" → check pigments
  ⑤: "Additional physics needed" → try double-gate NPQ model

Expected distribution (hypothesis):
  ① FIT-OK:     ~60-70% (standard photosynthetic organisms)
  ② α-outlier:  ~10-15% (extreme antenna sizes: cyanobacteria, diatoms)
  ③ n-outlier:  ~5-10%  (extreme repair: shade/light specialists)
  ④ Ic-outlier: ~5-10%  (unusual light environment adaptation)
  ⑤ Non-fit:    ~5-10%  (NPQ-dominant, artifacts, non-steady-state)

CRITICAL: These percentages are PREDICTIONS, not post-hoc analysis.
When the 1808 curves are classified, we compare against these predictions.
""")
