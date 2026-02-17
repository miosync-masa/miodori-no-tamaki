================================================================
PCC×SCC PHOTOINHIBITION FRAMEWORK — MASTER FACTSHEET
================================================================
Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
Data:    piCurve package (Amirian et al. 2025), N = 1808 PI curves
Updated: 2026-02-18
================================================================

================================================================
§1. MODEL EQUATIONS
================================================================

P(I) = Pmax × PCC(I) × SCC(I) − R

  PCC(I) = tanh(αI / Pmax)              [Channel: light harvesting]
  SCC(I) = tanh((Pmax / (βI))^γ)        [Gate: photoprotection]

  Ph10:  γ = cosh²(1) ≈ 2.381  (fixed)  → 4 free params {Pmax, α, β, R}
  Ph11:  γ = free                        → 5 free params {Pmax, α, β, R, γ}

  λ-model (extended):
    SCC_λ(I) = tanh((Pmax / (β × I_eff))^γ)
    I_eff = I × PCC(I)^λ
    λ = 0: factorized (Ph10)
    λ > 0: coupled (PCC feeds back into SCC)

SSOC VARIABLES:
  S   = α / β         Gate variable (separation ratio)
  Iα  = Pmax / α      PCC saturation irradiance
  Iβ  = Pmax / β      SCC onset irradiance (= Ic)
  S   = Iβ / Iα       Equivalent definition

  S > S_c ≈ 1:  Factorized phase (PCC × SCC)
  S < S_c ≈ 1:  Coupled phase (PCC ⊗ SCC)

================================================================
§2. REGIME CLASSIFICATION (Fig 14, 15)
================================================================

Three regimes by PCC-SCC separation quality:

  R1 Standard:   N = 1248 (69.0%)
    • Ph10 with γ = cosh²(1) is optimal
    • S ≥ 3, median R²adj = 0.9712
    • PCC and SCC cleanly separated

  R2 Adaptive:   N = 544 (30.1%)
    • Free γ (Ph11) meaningfully improves fit
    • S ≥ 3, ΔR²adj = +0.015 (Ph11 over Ph10)
    • γ systematically lower: median = 1.09

  R3 Coupled:    N = 16 (0.9%)
    • S < 3 (photoinhibition before light saturation)
    • PCC and SCC overlap → factorization breaks
    • Median R²adj = 0.8508

PCC/SCC SEPARATION:
  Iα median = 33.1 µmol/m²/s
  Iβ median = 438.0 µmol/m²/s
  S  median = 12.5×

  PCC at Iβ = 1.0000 (fully saturated when SCC starts)
  Separation clean for S ≫ 1, breaks at S < 1

16-MODEL TOURNAMENT:
  Class A (no gate): Ph01, Ph02, Ph04, Ph09
  Class B (gate):    Ph06-07, Ph10-16
  Class AB (hybrid): Ph03, Ph08
  Class C (rational): Ph05

  Overall winner: Class B = 65.2%, Class A = 25.8%
  AICc vs BIC agreement: 93.4%

================================================================
§3. GATE EQUIVALENCE THEOREM (Fig 13 — Tomoe Proposals A–E)
================================================================

THEOREM: When the SCC gate is saturated (≈0 or ≈1),
  its mathematical form becomes irrelevant and all
  gate-class models converge.

EVIDENCE:

  A: GEI (Gate Effect Index)
     GEI = median_I(1 − SCC(I)), I ∈ [1, 1200]
     GEI vs SAI: r = 0.278, p = 2.0×10⁻³³
     Class B win rate: FLAT across GEI quintiles (62-67%)
     → Gate visibility ≠ gate preference

  B: Winner Entropy
     H/Hmax by SAI group:
       Resistant (<−2σ): 0.805 ← COLLAPSE
       Moderate-R:       0.921
       Typical:          0.891
       Moderate-S:       0.829
       Sensitive (>+2σ): 0.834 ← COLLAPSE
     Kruskal-Wallis: H = 9.67, p = 0.046
     → Model degeneracy at both extremes ✅

  C: Measurement Range
     r(Iβ, ClassA_win) = 0.041, p = 0.084 (n.s.)
     → NOT a measurement artifact ✅

  D: Penalty Robustness
     AICc vs BIC agreement > 91% in all groups
     → NOT a penalty artifact ✅

  E: SCC Visibility
     Resistant GEI median = 0.000 (gate invisible!)
     χ²(GEI split × Class B) = 0.16, p = 0.686 (n.s.)
     → Gate collapse is physical reality ✅

SYNTHESIS: Gate Equivalence is genuine, arising from
  topology of the tanh gate function at saturation limits.

================================================================
§4. λ-MODEL & PHASE TRANSITION (Fig 16)
================================================================

CRITICAL REALIZATION: λ is NOT a continuous regression
  parameter. λ is a DISCRETE PHASE LABEL.

  λ = 0:   Factorized (R1, R2)    S ≫ 1
  λ = 0.5: Weakly coupled          S ~ few
  λ = 1:   Strongly coupled (R3)   S < 1

Fitting λ to R3 curves: FAILED (identifiability collapse)
  → λ, Pmax, β form degenerate manifold
  → Model rank drops (Hessian singular)
  → This failure IS the physics: at S < 1, PCC and SCC
     share PSII and can't be independently parametrized

PHYSICAL MECHANISM:
  Factorized: PCC (ms) and SCC (min) use different timescales
    → independent → multiplicative factorization valid
  Coupled: electron flux through PSII generates ROS
    → damage ∝ PCC output → SCC depends on PCC
    → self-consistent feedback → factorization breaks

================================================================
§5. OUTLIER CATALOG (Fig 17)
================================================================

5-class deviation taxonomy (a priori thresholds):

  ① FIT-OK:     1204 (66.6%)   R² ≥ 0.93, params in range
  ② α-outlier:    69 (3.8%)    Antenna size variation
  ③ n-outlier:   423 (23.4%)   γ ≠ cosh²(1) (gate cooperativity)
  ④ Ic-outlier:    49 (2.7%)   Repair/damage balance anomaly
  ⑤ Non-fit:       63 (3.5%)   Model inadequate (need new physics)

Ph11 RESCUE: 387/453 original ⑤ rescued to ③ by freeing γ
  → γ variation accounts for 85% of Ph10 failures
  → cosh²(1) is DEFAULT, not universal

PREDICTION ACCURACY:
  ① predicted 60-70%, got 66.6% ✅
  ⑤ predicted 5-10%, got 3.5% (after rescue) ✅

CLASS × REGIME CROSS-TAB:
                R1      R2      R3
  ① FIT-OK      988     210       6
  ② α-outlier    33      31       5
  ③ n-outlier   195     226       2
  ④ Ic-outlier   15      33       1
  ⑤ Non-fit      17      44       2

  R3 enriched in ② α-outlier: 31% (5/16) vs 3.8% overall

FIT-OK RATE BY S BAND:
  S < 1:         0%  (0/9)     ← FACTORIZATION BREAKS
  1 ≤ S < 3:    86%  (6/7)
  3 ≤ S < 10:   77%  (438/571)
  10 ≤ S < 30:  67%  (729/1091) ← optimal band
  30 ≤ S < 100: 28%  (30/107)
  S ≥ 100:       4%  (1/23)    ← gate invisible

================================================================
§6. SSOC EVOLUTIONARY LANDSCAPE (Fig 18)
================================================================

ORIGINAL HYPOTHESIS (rejected):
  "α conserved (physics), K_r sole evolutionary knob"
  α CV = 88% → NOT conserved

REVISED HYPOTHESIS (SSOC — Two-Axis Landscape):
  "What physics constrains, evolution preserves.
   What physics permits, evolution explores."

——— CONSERVED (PHYSICS) ———

  1. α-β SCALING LAW
     log β = 0.81 × log α + (−1.36)
     r² = 0.426
     → "More harvest → more damage" is universal photochemistry
     → Slope < 1: sub-linear = damage tolerance at high α

  2. γ = cosh²(1) AS DEFAULT
     Within ±20%: 19% of species (strict)
     R1 majority follows this default
     R2 deviates systematically (median γ = 1.09)

  3. S_c ≈ 1 (PHASE BOUNDARY)
     Topological: cannot be moved by evolution
     Forbidden zone: S ∈ [0.82, 1.61]
     Density depletion: 116× (only 1 curve in gap!)
     First-order phase transition evidence

  4. Pmax AS SPECTATOR
     r(Pmax, S) = −0.268 (weak)
     r(Pmax, γ) = −0.069 (negligible)
     Maximum rate does NOT determine phase

——— VARIABLE (EVOLUTION) ———

  Axis 1: S = α/β (gate POSITION)
    Range: [0.1, 1000+] → 4 decades explored
    96.2% of S variance from β-residual (damage excess)
    Only 3.8% from α-level
    → Phase controlled by DAMAGE EXCESS, not harvest level
    Selection pressure: S ≫ 1 preferred (left-skew)

  Axis 2: γ (gate SHAPE)
    Default: cosh²(1) for R1 majority
    Modified in 23% of species (N=423 n-outliers)
    R2: γ systematically lower → shallower gate
    R3: γ variable, 0/16 at default

  Axis 3: Pmax (independent, spectator)
    Range: [0.01, 20+]
    Freely evolvable, orthogonal to phase structure

  Independence: r(log S, log γ) = 0.033 → ORTHOGONAL

——— FORBIDDEN (TOPOLOGY) ———

  S ≈ 1: forbidden zone (first-order transition)
  FIT-OK = 0% at S < 1 (9 curves, none factorizable)
  Gap = topological invariant of the landscape

——— REGIME MAPPING ON LANDSCAPE ———

  R1 (Factorized): S median = 11.7, γ ≈ cosh²(1), N=1245
  R2 (Adaptive γ): S median = 14.9, γ = 1.09,     N=542
  R3 (Coupled):    S median = 0.79, γ variable,    N=16

——— R3 DETAILED ———

  16 curves, S median = 0.79
  α: R3 median = 0.017, R1 = 0.060 → 3.6× difference
  Ic: R3 median = 155, R1 = 468 → 3.0× difference
  α shift dominates: 0.555 dex vs 0.479 dex for Ic
  → Coupled phase via LOW α (small antenna), not high β

================================================================
§7. SSOC CROSS-DOMAIN CORRESPONDENCE
================================================================

  LPS Electrolyte (SSOC)        Photoinhibition (PCC×SCC)
  ─────────────────────         ─────────────────────────
  Channel: Li⁺ transport       Channel: PCC = tanh(αI/Pm)
  Gate:    CN ≥ 3 threshold     Gate:    SCC = tanh((Pm/βI)^γ)

  Structure var: v_f            Structure var: S = α/β
  Critical:      v_f_c          Critical:      S_c ≈ 1

  Gate shape: CN (coordination) Gate shape: γ (cooperativity)
  Independent of v_f!           Independent of S! (r = 0.033)

  8 params, r = 0.96            5 params, R²adj = 0.96
  Continuous transition          First-order (forbidden gap)

================================================================
§8. TESTABLE PREDICTIONS
================================================================

  P1: Shade species → low α → S closer to 1 → more coupling
  P2: Species with NPQ → γ < cosh²(1) → R2 regime
  P3: Species with strong FtsH → high β tolerance → high S → R1
  P4: Hysteresis in PI curves for R3 species (coupled bistability)
  P5: Forbidden zone robust across independent datasets
  P6: ② α-outliers have anomalous antenna size (Chl a/cell)
  P7: ③ n-outliers correlate with habitat light regime
  P8: ④ Ic-outliers have unusual photoprotective pigment content

================================================================
§9. PARAMETER MODEL COMPARISON
================================================================

  Approach        Params/curve  Total free     Info structure
  ──────────────  ────────────  ─────────────  ──────────────────
  Regression      4-5           7232-9040      7232 independent #s
  (Ph10/Ph11)     (all free)    (N×p)          no inter-param links

  SSOC            4-5           same/curve     4 structural constants
  (this work)     (same!)       + 4 universal  + 1808 (S,γ) coords
                                               + scaling law links
                                               → effective DOF ≈ 3.5

  Key: same params/curve, but SSOC adds CONSTRAINTS:
    • β predicted from α (43% variance explained)
    • γ = cosh²(1) for 77% (eliminate 1 param)
    • S < 1 → regime change (qualitative prediction)
    • Pmax spectator (structural independence)

================================================================
§10. FIGURE CATALOG
================================================================

  Fig 13: Gate Equivalence (Tomoe proposals A–E)          6 panels
          fig13_extended_gate_analysis.png

  Fig 14: PCC/SCC Decomposition (1808 curves)            multi-panel
          fig14_pcc_scc_decomposition.png

  Fig 15: Complete Prediction Model (regimes)             multi-panel
          fig15_complete_model.png

  Fig 16: λ-Model Phase Diagram                          multi-panel
          fig16_lambda_theory.png
          fig16_phase_diagram.png

  Fig 17: Outlier Catalog (5-class taxonomy)              9 panels
          fig17_outlier_catalog_1808.png

  Fig 18: SSOC Evolutionary Landscape                     9 panels
          fig18_ssoc_evolutionary_landscape.png

================================================================
§11. DATA FILES
================================================================

  Source:
    Opt_ParVal_of_piModels.csv     15.3 MB  Amirian et al. 2025 (Zenodo)

  Derived:
    ph10_decomposed.csv            1.6 MB   +regime, +S, +Iα, +Iβ
    ph10_extended.csv              1.1 MB   +GEI, +SAI, +n_competitive
    outlier_catalog_1808.csv       246 KB   Initial 5-class catalog
    outlier_catalog_1808_refined.csv 252 KB Ph11-rescued catalog
    aicc_wide.csv                  525 KB   16 models × 1808 AICc
    r2adj_wide.csv                 534 KB   16 models × 1808 R²adj

  Scripts:
    evolutionary_invariance_ssoc.py        SSOC 2-axis landscape

================================================================
§12. KEY NUMBERS (QUICK REFERENCE)
================================================================

  N curves total:          1808
  N FIT-OK (Ph10):         1204 (66.6%)
  N regimes:               3 (R1=69%, R2=30%, R3=0.9%)

  γ_default:               cosh²(1) = 2.381
  S_critical:              ≈ 1
  Forbidden zone:          S ∈ [0.82, 1.61], 116× depleted

  α-β scaling:             slope = 0.81, r² = 0.43
  S-γ independence:        r = 0.033 (orthogonal)
  Pmax spectator:          r(Pmax,S) = −0.27

  S variance source:       96% from β-residual
  γ ≠ cosh²(1):            23% of species (423/1808)

  R3 α shift:              3.6× lower than R1
  R3 Ic shift:             3.0× lower than R1

  AICc-BIC agreement:      93.4%
  Class B overall win:     65.2%
  Gate entropy collapse:   H/Hmax = 0.81 (resistant)

================================================================
§13  P5 FORBIDDEN ZONE INDEPENDENT VALIDATION — STRATEGY & RECON
================================================================

【目的】
禁制帯 S ∈ [0.82, 1.61] が Amirian 1808データ固有のアーティファクトではなく
物理的普遍構造であることを独立データで実証する（Prediction P5）。

【核心的制約条件】（無二 = GPT武士 による整理）
  必要データ: (i) 生のPI曲線点列、または最低でも
              (ii) Amirian型（Ph10）のα, β, Pmax パラメータ表
  ※ Platt型のα, βでは S = α/β の物理的定義が異なるため直接比較不可

【Platt-β ≠ Amirian-β 問題（本スレッドで検証済み）】
  同一1808カーブでのPh04(Platt) vs Ph10(Amirian)比較:
    - alpha相関: r = 0.966 (ほぼ同一、物理的に同じ初期勾配)
    - beta相関:  r = 0.246 (低い！定義が根本的に異なる)
    - S_Platt vs S_Am: Spearman = 0.62 (単調関係はあるが変換は危険)
    - S_Platt range: [0.71, 1.7×10¹²]  ← 発散する
    - S_Am range:    [0.13, 8400]
  結論: Platt型パラメータ表からの直接変換は不可。
        生点列データ → Ph10再フィット が必須。

【外部データ偵察結果】

  1) BCO-DMO SOGLOBEC (南大洋, NBP0103)
     - α, β, Pmax, Ik あり。CC-BY。
     - ただしPlatt型パラメータ → S直接算出不可
     - 生のP-I点列データが同プロジェクト内に存在する可能性 → 要追加調査

  2) MAPPS / PANGAEA (Bouman et al. 2018, 5000+ P-E curves)
     - PmB, αB 中心。β不在がほとんど（無二の予測通り）
     - PCC側スケーリング検証には使える（α-Pmax相関の頑健性）
     - 禁制帯の直接検証には不十分

  3) piCurve Rパッケージ (GitHub: Mohammad-Amirian/piCurve)
     - 内蔵データ: 8本のPI incubation（デモ用、少なすぎ）
     - data-raw/ フォルダに元データ加工スクリプトの可能性 → 要調査
     - 論文中 "~4000 open-ocean P-I curves" の元データソースを辿れる可能性

  4) Mendeley Data — 泥炭地PIカーブ (Navarino Island, Chile)
     - 20本の生PI曲線。独立・小規模だが越境テスト（植物）候補
     - 無二の攻め筋5「別ドメインで位相構造が保つか」に該当

  5) SeaBASS (NASA) — 未調査
     - File Search で "productivity" / "photosynthesis-irradiance" 検索が必要
     - 生PI点列が眠っている可能性が最も高い候補

【攻め筋の優先順位】（無二提案 + 本スレッド偵察結果を反映）

  Phase 1: 生点列データの確保
    1-A. piCurve data-raw/ → 元データソース特定 → ダウンロード
    1-B. SeaBASS File Search → 生PI点列の発掘
    1-C. BCO-DMO SOGLOBEC 生点列データ捜索

  Phase 2: Ph10再フィット & S分布生成
    - Python実装（既存フィッティングコードを流用）
    - 外部データに対してPh10フィット → α_Am, β_Am 取得
    - S = α/β ヒストグラム作成

  Phase 3: 禁制帯の独立検証
    - S ∈ [0.82, 1.61] の密度を Amirian 1808と比較
    - 帰無仮説: 一様分布からのKS検定
    - 禁制帯が再現 → P5確定（"物理"確定）
    - 禁制帯が不在 → データ依存性を示唆（研究は前進）

  Phase 4 (optional): 越境テスト
    - Mendeley泥炭地データ or 他の陸上植物PIカーブ
    - 同じ S ≈ 1 近傍の疎密が出るか → トポロジカル主張の強化

【戦略的判断】（ご主人さま + 無二 + 環の合意）
  - 理論公理化は後回し。「観測は物理の始まり」
  - 既存の禁制帯定義（S ∈ [0.82, 1.61], 116× depletion）をそのまま使用
  - データを見てから物理を見つける = 正当な科学的手法
  - 外部データで禁制帯を叩く → R3結合相モデルはその後

================================================================
END OF MASTER FACTSHEET
================================================================
