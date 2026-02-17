# PCC×SCC Photoinhibition Framework — Complete Factsheet
# Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
# Date: 2026-02-17
# Dataset: Amirian et al. 2025, 1808 PI curves with photoinhibition
# Zenodo: https://doi.org/10.5281/zenodo.16748102

================================================================
## CORE FRAMEWORK
================================================================

Model: P(I) = Pmax × tanh(αI/Pmax) × tanh((Pmax/βI)^γ) - R
       γ = cosh²(1) ≈ 2.38

Topology: P(I) = PCC(I) × SCC_Gate(I)
  PCC = tanh(αI/Pmax)          — light harvesting (Percolation Connected Component)
  SCC = tanh((Pmax/βI)^γ)      — photoinhibition gate (Sub-Critical Component)

Key derived quantities:
  Iα = Pmax/α    — light saturation irradiance
  Iβ = Pmax/β    — photoinhibition onset irradiance
  Plateau width = log10(Iβ/Iα)

SAI (SCC-Adaptation Index):
  log β = 0.806 × log α + (intercept)   [regression on 1808 curves]
  SAI = residual of log β from log α regression
  SAI < 0 → more resistant than expected (strong SCC defense)
  SAI > 0 → more sensitive than expected (weak SCC defense)

================================================================
## HYPOTHESIS 1: SAI LEFT-SKEWNESS = NATURAL SELECTION FINGERPRINT
## STATUS: ✅ CONFIRMED
================================================================

Bootstrap analysis (B=10,000, N=1,808):

  Skewness γ₁ = -1.4346
  Bootstrap mean: -1.4065
  95% CI: [-2.297, -0.485]
  99% CI: [-2.539, -0.211]  ← EXCLUDES ZERO
  99.9% of bootstrap samples negative
  D'Agostino test: z = -18.96, p = 3.68×10⁻⁸⁰

Kurtosis: +12.35, 95% CI: [+6.93, +17.78] (leptokurtic)

Tail asymmetry:
  Left tail (< -2σ): 57 curves (3.15%)
  Right tail (> +2σ): 17 curves (0.94%)
  Ratio: 3.35× more extreme resistant organisms

Quantile asymmetry |Left|/|Right|:
  Q1 vs Q99: 1.85×
  Q5 vs Q95: 1.40×
  Q10 vs Q90: 1.24×
  → Asymmetry increases toward tails

Extreme values:
  1st percentile: -0.998 (resistant)
  99th percentile: +0.580 (sensitive)
  |P1|/P99 = 1.72

KS vs Gaussian: KS=0.089, p=9.3×10⁻¹³

Most extreme resistant: PI002120 (SAI = -2.80, β = 7×10⁻⁶, 800× below mean)
Top 1% resistant (N=18): β range 10⁻⁵ to 10⁻⁴, R²adj mean = 0.882

Interpretation: Directional selection under high-light → evolve strong SCC 
repair (FtsH, NPQ) or die → survivorship tail in resistant direction.

Testable prediction: SAI left-skewness should be MORE pronounced in 
high-light environments (surface ocean, tropics).

Figure: fig10_selection_fingerprint.png (6 panels)

================================================================
## HYPOTHESIS 2: SAI = TOTAL DEFENSE PROXY (NPQ + REPAIR)
## STATUS: ✅ CONFIRMED
================================================================

STRATEGY 1: Cross-model R² gap (Amirian − Platt)

  SAI vs R²_gap: r = -0.6514 (p = 7.98×10⁻²¹⁹)  ← EXTREMELY STRONG

  By SAI group:
    Resistant (SAI < -2σ): median gap = +0.141, N=57
    Typical:               median gap = +0.054, N=1734
    Sensitive (SAI > +2σ): median gap = +0.004, N=17
  
  Mann-Whitney Resistant vs Typical: p ≈ 0 (SIGNIFICANT)
  
  Meaning: Resistant curves have WIDER plateaus → Platt model (no plateau)
  fails more → larger gap → SAI captures plateau width = total defense

STRATEGY 2: Pmax × SAI interaction

  Pmax vs SAI: r = +0.3877 (p = 6.33×10⁻⁶⁶)

  SAI skewness by Pmax quartile:
    Q1 (low Pmax):  skew = -1.54 [CI: -2.62, -0.07] ✅ significant
    Q2:             skew = -2.11 [CI: -2.55, -1.47] ✅ significant
    Q3:             skew = +0.03 [CI: -0.93, +1.02] ❌ not significant
    Q4 (high Pmax): skew = +1.74 [CI: -0.10, +2.79] ❌ not significant

  Resistant fraction by Pmax:
    Q1: 8.2%, Q2: 4.0%, Q3: 0.2%, Q4: 0.2%

  Key finding: Selection fingerprint (left-skew) appears ONLY in low-Pmax
  organisms. High-Pmax organisms use capacity strategy, not repair.

STRATEGY 3: 2D ecological space (α* vs SAI)

  α* = residual(log α | log Pmax)  [absorption independent of capacity]
  α* vs SAI: r = -0.3241 (p = 1.68×10⁻⁴⁵)
  → High absorbers are more resistant

  Four ecological strategies:
    Absorb+Repair:      27.2%, med_Pmax=1.25
    Absorb (vulnerable): 20.5%, med_Pmax=2.17
    Avoid+Repair:        17.5%, med_Pmax=1.47
    Avoid (passive):     34.8%, med_Pmax=2.53  ← DEFAULT strategy

Figure: fig11_mechanism_hypothesis2.png (6 panels)

================================================================
## HYPOTHESIS 3: (α*, SAI) 2D ECOLOGICAL SEPARATION
## STATUS: ✅ CONFIRMED (within Hypothesis 2 analysis)
================================================================

  α* vs SAI: r = -0.324 (p = 10⁻⁴⁵)
  Two independent ecological axes confirmed
  Four-quadrant strategy space with distinct Pmax profiles

================================================================
## HYPOTHESIS 4: NON-FIT = LOW-Pmax STATISTICAL ISSUE
## STATUS: ✅ CONFIRMED (from Phase 2)
================================================================

  Non-fit curves (β ≤ 0 or convergence failure):
    Pmax median: 0.556
  Good-fit curves:
    Pmax median: 1.934
  
  Mann-Whitney: p = 4.3×10⁻¹⁰
  
  Low Pmax → poor signal/noise → parameter identifiability collapse
  Model failure is statistical, not biological

================================================================
## HYPOTHESIS 5: GATE FUNCTION EQUIVALENCE CLASSES
## STATUS: ✅ CONFIRMED (with predictive reversal → deeper physics)
================================================================

GATE CLASSIFICATION:

  Class A (Immediate decay, no plateau):
    Ph01 Platt 1980, Ph02 Steele 1962, Ph04 Neale 1987, Ph09 Michaelis×exp
  
  Class AB (Hybrid):
    Ph03 tanh×exp, Ph08 exp×tanh(1/I)ᵧ
  
  Class B (Reciprocal saturating, explicit plateau):
    Ph06, Ph07, Ph10 Amirian, Ph11 Amirian-γ, Ph12 tanh×tanh-sym,
    Ph13 exp-tanh-γ, Ph14, Ph15, Ph16
  
  Class C (Rational/algebraic):
    Ph05 Eilers-Peeters

AICc TOURNAMENT (1808 curves):

  Class B wins: 1178/1808 = 65.2%
  Class A wins:  463/1808 = 25.6%
  Class C wins:   90/1808 =  5.0%
  Class AB wins:  77/1808 =  4.3%

  Top 3 individual models:
    Ph10 (Amirian):    278 wins (15.4%)
    Ph09 (Michaelis×exp): 260 wins (14.4%)
    Ph12 (tanh×tanh-sym): 249 wins (13.8%)

SAI × GATE PREFERENCE (key finding):

  Gate Class B fraction by SAI group:
    Resistant (2σ): 49.1%  ← LOWER than typical (逆！)
    Typical:        67.6%  ← Class B dominates here
    Sensitive (2σ): 47.1%  ← Also lower

  χ²(Resistant vs Typical) = 9.21, p = 0.027 (significant)

  ΔAICc(Platt−Amirian) vs SAI: r = -0.651 (reinforces H2)

GATE EQUIVALENCE THEOREM:

  "Gate function selection matters ONLY in the transition region
   where the gate value is between 0 and 1.
   
   When gate ≈ 1 (Resistant: β→0, no inhibition visible):
     All gate functions collapse → Class A ≅ B ≅ C
   
   When gate ≈ 0 (extreme inhibition, curve collapses):
     All gate functions collapse → same result
   
   When gate ∈ (0,1) (Typical: inhibition is moderate):
     Gate SHAPE matters → Class B wins (plateau-capable)"

  Analogy to LPS:
    v_f << v_c: p_active ≈ 1, Hill gate shape invisible
    v_f ≈ v_c:  p_active transitions, Hill n=3 matters
    v_f >> v_c: p_active ≈ 0, all models predict zero

EMPIRICAL CLUSTERING (AICc rank Spearman correlation):

  16 models → 3 empirical clusters:
    Cluster 1: Ph02,04,05,06,14,15 (mixed A/B/C)
    Cluster 2: Ph09,10,12 (Amirian neighborhood)
    Cluster 3: Ph01,03,07,08,11,13,16 (Platt neighborhood + shape models)

  Note: Empirical clusters ≠ a priori gate classes
  → Data-driven behavior grouping is more fundamental than
     mathematical form classification

Pmax × GATE PREFERENCE:
    Q1 (low Pmax):  B=66.8%, A=20.6%
    Q2:             B=67.9%, A=23.7%
    Q3:             B=64.6%, A=26.3%
    Q4 (high Pmax): B=61.3%, A=31.9%
  → Slight trend: high-Pmax organisms are more "model-agnostic"

Figure: fig12_gate_equivalence.png (6 panels)

================================================================
##巴の拡張提案（未実装 — 次フェーズ用）
================================================================

A案: Gate Effect Index (GEI) 定義
  GEI = median_I(1 - P(I)/P_PCC(I))
  予測: Class B勝率 vs GEI は「山型」（中域で最大、両端で縮退）

B案: 勝者エントロピー
  H = -Σ p_m log p_m  per SAI bin
  予測: SAI両端でH低下（縮退）、中央でH上昇

C案: 測定レンジ（Imax不足）検証
  Imax/Iβ比で層別 → Class A増加がImax不足に起因か確認

D案: AICc罰則チェック
  AIC/BIC/CVを併用 → 端の群で勝者が揺れれば罰則起因

E案: SCC実装差の検証
  Resistant群をGEI小/GEI中に二分
  → GEI中でClass B勝ち、GEI小でClass A勝ちなら「可視性」確定

================================================================
## KEY NUMBERS FOR MANUSCRIPT (QUICK REFERENCE)
================================================================

N = 1808 curves (photoinhibition subset)
50 years of DFO data (1973-2022), 1304 locations

SAI distribution:
  mean = 0, σ = 0.303, skew = -1.435, kurtosis = 12.39

H1 headline: γ₁ = -1.435, 99%CI [-2.539, -0.211], p = 3.7×10⁻⁸⁰
H2 headline: r(SAI, ΔR²) = -0.651, p = 8.0×10⁻²¹⁹
H3 headline: r(α*, SAI) = -0.324, 4 ecological strategies
H4 headline: Pmax(non-fit) = 0.56 vs Pmax(fit) = 1.93, p = 4.3×10⁻¹⁰
H5 headline: Class B = 65.2%, χ²(SAI) = 9.21, p = 0.027

Regression: log β = 0.806 × log α + intercept (r² = 0.427)
Variance decomposition: 57.3% of β variation independent of α

================================================================
## FILES
================================================================

Code:
  fig10_selection_fingerprint.py  — H1 bootstrap + extreme values + 6-panel figure
  fig11_mechanism.py              — H2 cross-model gap + Pmax interaction + 6-panel
  fig12_gate_equiv.py             — H5 AICc tournament + gate classes + 6-panel

Figures:
  fig10_selection_fingerprint.png — Natural selection fingerprint
  fig11_mechanism_hypothesis2.png — Mechanism evidence (NPQ proxy)
  fig12_gate_equivalence.png      — Gate equivalence classes

Data:
  amirian_params.csv              — 16 models × 1808 curves (from Zenodo)
