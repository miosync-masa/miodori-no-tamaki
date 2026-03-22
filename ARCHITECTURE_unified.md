# Midori-no-Tamaki: Unified Architecture
## EOS-Thermal Extension for Photobioreactor Monitoring & Control

**Authors:** M. Iizumi & T. Iizumi (Miosync, Inc.)
**Date:** 2025-03-22 (Session 1 consolidated)
**Status:** Theory established, cross-species validated, implementation-ready

---

## 1. Design Philosophy

The original EOS (Equation of State for PI curves) transforms photosynthesis–irradiance curves from fitted objects into predictable state functions. This extension applies the same philosophy to environmental stress: **predict the full photosynthetic state from minimal, real-time sensor readings using closed-form physics**.

**Core principle:** At any moment, one bottleneck dominates. The EOS identifies which one. The operator knows which knob to turn.

**Key constraint:** No differential equations in the state estimation layer. All dynamics are captured by sensors measuring the *current* state. Time evolution belongs to a separate control layer — or simply to continuous monitoring.

---

## 2. The Bottleneck-Minimum Formulation

```
β_eff(I, T, pH, DIC) = min(
    β_light(I),           # photon-induced D1 damage
    β_thermal(T),         # D1 repair thermodynamic limit
    β_carbon(pH, DIC, T)  # electron sink deficit from carbon limitation
)
```

Each term is a closed-form expression derived from physical constants. The dominant bottleneck determines the active regime. This is Liebig's Law of the Minimum expressed as a mathematical min-function.

### Connection to Original EOS

| Original EOS | Thermal Extension |
|---|---|
| β is a fitted parameter | β_eff is computed from physics |
| SAI = empirical residual | SAI = sum of identified stress components |
| S = α/β classifies 1D regime | (S, T, pH) classifies 3D regime |
| EOS2: (α, Pmax) → PI curve | EOS2-extended: (α, Pmax, T, pH) → PI curve |

---

## 3. Channel 1: β_light — Light Damage

**Physics:** Photon absorption by Mn cluster in oxygen-evolving complex → D1 protein damage.

**Key property:** Damage rate kd is proportional to light intensity and **temperature-independent**.

**Evidence:**
- Allakhverdiev & Murata (2004): kd = 0.200 ± 0.002 min⁻¹ at 2000 µE, across 10–34°C (CV = 0.9%)
- Torzillo & Vonshak (1994): kd constant across 25–40°C (our fit, CV = 20%)

**Closed form:**
```
kd(I) = σ_PSII × I
```
where σ_PSII is the effective PSII cross-section (photophysical, not enzymatic).

**Regime boundary:** The original EOS gate variable S = α/β determines when light damage becomes the dominant bottleneck.

---

## 4. Channel 2: β_thermal — Temperature-Dependent Repair

**Physics:** D1 repair requires: FtsH protease (damaged D1 degradation) → ribosomal psbA translation → new D1 synthesis → PSII reassembly. All steps are enzymatic and ATP-dependent.

**Key property:** Repair rate kr follows Arrhenius kinetics with a denaturation ceiling.

**Evidence (cross-species validated):**

| Source | Organism | Ea_repair | Denaturation |
|---|---|---|---|
| Allakhverdiev 2004, D1 synthesis | Synechocystis | 87 kJ/mol | — |
| Allakhverdiev 2004, repair rate | Synechocystis | 74 kJ/mol | — |
| Ueno 2016 | Synechocystis | — | Tm ≈ 42–44°C |
| Rehder 2023, acute Pmax shift | Phaeodactylum (diatom) | 48 kJ/mol | — |
| Literature default | General | 63 kJ/mol (0.65 eV) | Species-dependent |

**Note on Ea_r = 240 kJ/mol:** Our initial 2-point fit from Torzillo 1994 (25°C, 35°C) overestimated Ea_r because the 25°C data point included non-linear ROS-mediated translational collapse (Nishiyama 2001), not captured by simple Arrhenius. Independent multi-temperature data consistently yield 50–90 kJ/mol. We adopt 63 kJ/mol (0.65 eV) as the operational default, pending Spirulina-specific calibration.

**Closed form:**
```
kr(T) = kr0 · exp(-Ea_r / R·T_K) · 1/(1 + exp((T - Tm)/δT))
        |---- Arrhenius cold branch ----|  |-- denaturation sigmoid --|

Parameters:
  Ea_r  = 63 kJ/mol (0.65 eV, literature default)
  Tm    = 42°C (Synechocystis; Spirulina may be similar due to thermotolerance)
  δT    = 3°C (transition width)

β_thermal(T) = kr(T) / kr(T_opt)  (normalized to optimal)
```

**Cross-species universality confirmed:**
- α is temperature-independent in Synechocystis (Allakhverdiev 2004) AND Phaeodactylum (Rehder 2023, ratio = 1.01)
- kd is temperature-independent (CV = 0.9%, 10–34°C)
- Structure kd×I / kr(T) holds across prokaryotes and eukaryotes

**Operator significance:**
```
I_cross(T) = kr(T) / kd = critical irradiance where damage = repair

T = 25°C: I_cross ≈ low  → morning cold = photoinhibition risk
T = 35°C: I_cross ≈ high → operational light range is safe
```

---

## 5. Channel 3: β_carbon — Carbon Supply Limitation

**Physics:** When CO₂/HCO₃⁻ supply to RuBisCO is insufficient, the Calvin cycle slows, electrons accumulate on PSI acceptor side, producing ROS, which inhibits D1 repair (kr) — creating an indirect coupling to β_thermal.

### 5.1 Carbonate Equilibrium (closed form, well-established chemistry)

```
CO₂(aq) ⇌ HCO₃⁻ ⇌ CO₃²⁻

DIC = [CO₂] + [HCO₃⁻] + [CO₃²⁻]

α₀(pH) = CO₂ fraction  = 1 / (1 + K1/H + K1·K2/H²)
α₁(pH) = HCO₃⁻ fraction = 1 / (H/K1 + 1 + K2/H)
α₂(pH) = CO₃²⁻ fraction = 1 / (H²/(K1·K2) + H/K2 + 1)

K1(T) ≈ 4.3×10⁻⁷ at 25°C (temperature-dependent, tabulated)
K2(T) ≈ 4.7×10⁻¹¹ at 25°C (temperature-dependent, tabulated)
```

### 5.2 Effective Carbon Concentration

Spirulina uses HCO₃⁻ via CCM (Carbon Concentrating Mechanism), not CO₂(aq) directly.
```
Ci_eff(pH, DIC) = DIC × α₁(pH)    [mM HCO₃⁻ available]

Zarrouk medium: DIC ≈ 200 mM (16.8 g/L NaHCO₃)
  pH 9.5:  Ci_eff ≈ 180 mM (HCO₃⁻ dominant, ~90%)
  pH 12:   Ci_eff ≈ 20 mM  (CO₃²⁻ dominant, HCO₃⁻ depleted)
  pH 12+:  Ci_eff → 0       (growth stops — Kobayashi 1996)
```

### 5.3 β_carbon Definition (Michaelis-Menten, closed form)

```
β_carbon(pH, DIC, T) = Ci_eff / (Km_eff + Ci_eff)

Km_eff: effective half-saturation for HCO₃⁻ uptake (~5 mM after CCM correction)
```

### 5.4 Experimental Evidence

Kobayashi & Fujita (1996), Spirulina NIES-39, 25°C, 250 µE/m²/s:

| Condition | pH range | Max biomass | Growth stopped by |
|---|---|---|---|
| No pH control | 8.95 → 12+ | 1.2 g/L | CO₃²⁻ dominates, HCO₃⁻ gone |
| pH controlled (CO₂) | 8.5–10 | 2.2 g/L | Nitrogen depletion |
| pH + N controlled | 8.5–10 | 4.2 g/L | Mutual shading (light) |

This demonstrates **sequential bottleneck transitions**: β_carbon → N-limitation → β_light.

---

## 6. pH–CO₂–Light–Temperature Interactions

### 6.1 The Physical Chain

```
CO₂ dissolves → H⁺ released → pH decreases
Photosynthesis consumes CO₂ → H⁺ removed → pH increases
CO₂ gas supply → replenishes CO₂ → pH decreases (control mechanism)
Temperature up → CO₂ solubility down → less CO₂ per unit gas supply
Temperature up → enzyme rates up → faster CO₂ consumption → pH rises faster
```

### 6.2 Feedback Loop

```
High light → fast Pgross → rapid CO₂ consumption → pH rises → 
→ HCO₃⁻ decreases → β_carbon drops → Pgross limited → CO₂ consumption slows → 
→ pH stabilizes (NEGATIVE FEEDBACK within pH 8.5–10.5)

BUT: if pH exceeds ~11, HCO₃⁻ → CO₃²⁻ transition accelerates → 
→ RUNAWAY to pH 12+ → irreversible growth stop
```

### 6.3 Interaction Matrix

```
              Light(I)    Temp(T)     CO₂/pH      Nitrogen(N)
β_light       DIRECT      —           indirect    —
β_thermal     indirect    DIRECT      indirect    —
β_carbon      indirect    indirect    DIRECT      —
η(allocation) —           DIRECT      —           DIRECT
```

Each channel has ONE primary control knob. Cross-talk occurs through feedback loops but does not change the primary control structure.

### 6.4 Temperature Trade-off

Temperature increase simultaneously:
1. **Improves** β_thermal (kr↑, Arrhenius)
2. **Worsens** β_carbon (CO₂ solubility↓, Henry's law)
3. **Increases** respiration (R(T)↑, nighttime biomass loss↑)
4. **Increases** CO₂ consumption rate (faster pH rise)

→ **Higher temperature requires more aggressive CO₂ supply.** The optimal operating point is not temperature alone but (T, CO₂_supply) jointly.

---

## 7. SAI Decomposition

The original SAI = log₁₀(β_obs) − log₁₀(β_pred(α)) now has physical meaning:

```
SAI_total = SAI_thermal(T)           # Arrhenius, Ea ≈ 65 kJ/mol
          + SAI_carbon(pH, DIC)      # Michaelis-Menten, pH-dependent
          + SAI_nitrogen(N_status)   # D1 synthesis raw material limitation
          + SAI_ROS(I_history, T)    # Accumulated oxidative stress
```

Each component is independently measurable and interpretable. The original piCurve dataset's SAI scatter (σ = 0.303) reflects the **sum of all unmeasured environmental conditions** — temperature, pH, nutrients, light history — varying across 1,808 curves.

**Insight from today:** Ea_r measured values "scatter" across papers (42–87 kJ/mol) because cultivation conditions modulate kr through multiple pathways (ROS, ATP, salt, N). The base Arrhenius slope is ~63 kJ/mol; deviations are SAI_ROS, SAI_carbon, SAI_nitrogen overlaid on it.

---

## 8. Monitoring Architecture

### 8.1 Sensor Requirements

| Sensor | Measures | Feeds | Cost |
|---|---|---|---|
| pH probe | pH(t) | β_carbon | ~$5 |
| Thermometer | T(t) | β_thermal | ~$5 |
| PAM (optional) | α, Pmax | β_light, S regime | ~$10k+ |
| DIC (from medium recipe) | DIC₀ | β_carbon | Free |

**Minimum viable system: pH + Temperature.** PAM adds precision but is not required for bottleneck diagnosis.

### 8.2 Computation (all closed-form, instant)

```python
# Inputs: pH, T, DIC (known from medium)
Ci_eff = DIC * alpha1(pH, T)                        # carbonate equilibrium
b_carbon = Ci_eff / (Km_eff + Ci_eff)               # Michaelis-Menten
b_thermal = exp(-Ea_r/R*(1/T_K - 1/T_opt_K))        # Arrhenius
            * sigmoid(Tm, T, dT)                     # denaturation
b_eff = min(b_thermal, b_carbon)                     # bottleneck
```

No differential equations. No time integration. No ML model. **Pure physics, evaluated at the current sensor readings.**

### 8.3 Operator Decision Layer

```
IF β_carbon < β_thermal:
    → "Carbon limited: increase CO₂ supply, check pH"
IF β_thermal < β_carbon:
    → "Temperature limited: warm culture toward 35°C"
IF both OK but growth plateaus:
    → "Check nitrogen, or harvest (light-limited by density)"

ALERTS:
    pH > 10.5   → "CO₂ urgently needed"
    pH > 12     → "EMERGENCY: culture dying"
    T < 28°C    → "D1 repair severely impaired"
    T > 40°C    → "Denaturation risk"
```

### 8.4 Layer Structure

```
Layer 0: Sensors       → pH, T, (PAM optional)        [measure]
Layer 1: EOS (closed)  → β_eff, bottleneck ID         [compute instantly]
Layer 2: Threshold     → Alarms, operator guidance     [rule-based]
Layer 3: Optimization  → Optimal (I, T, CO₂) for      [ML or model-based,
                         target objective function       future work]
```

Layers 0–2 are implementation-ready today. Layer 3 is future.

---

## 9. Objective Functions (Application-Dependent)

Same 3 knobs (I, T, CO₂), different optimal settings:

```
J_CO2_absorption = Pgross(I,T,CO₂) - R(T)×night_fraction
    → Maximizes net carbon fixation (carbon credit applications)
    → Favors lower night temperature (reduces R(T))

J_growth = Pgross × η_protein(T) - R(T)
    → Maximizes biomass production
    → Requires T ≈ 35°C for optimal allocation

J_protein = Pgross × η_protein(T) × protein_fraction(T, N)
    → Maximizes protein yield (food security / "chicken replacement")
    → Requires T ≈ 35°C AND adequate nitrogen supply
```

The EOS provides the state estimation; the objective function determines which optimum to target.

---

## 10. Physical Constants Summary

### Confirmed (literature + independent validation)

| Constant | Value | Source | Status |
|---|---|---|---|
| kd temperature independence | CV = 0.9% (10–34°C) | Allakhverdiev 2004 | CONFIRMED |
| Ea_repair (operational default) | 63 kJ/mol (0.65 eV) | Literature consensus | ADOPTED |
| Ea_repair (Synechocystis range) | 48–87 kJ/mol | Allakhverdiev 2004, Ueno 2016, Rehder 2023 | CONFIRMED |
| Denaturation Tm | 42–44°C (Synechocystis) | Ueno 2016 | CONFIRMED |
| D1 synthesis = repair bottleneck | Confirmed | Allakhverdiev 2004, 2005 | CONFIRMED |
| ATP required for repair | DCCD abolishes repair | Allakhverdiev 2005 | CONFIRMED |
| α temperature independence | Ratio ≈ 1.01 (6°C vs 15°C) | Rehder 2023 | CONFIRMED |
| Ea_respiration | 48.8 kJ/mol | Torzillo 1994 | CONFIRMED |
| R(T) equation | 0.771·exp(0.0616T) | Torzillo 1994 | CONFIRMED |
| pH growth stop | pH 12+ | Kobayashi 1996 | CONFIRMED |
| γ₀ | cosh²(1) ≈ 2.381 | Original EOS (Iizumi 2026) | KNOWN |
| α–β scaling (m, c) | 0.814, −1.355 | Original EOS | KNOWN |
| Design law k | 50.4 | Original EOS | KNOWN |

### Adopted (reasonable defaults, refinable)

| Constant | Value | Basis | Status |
|---|---|---|---|
| Km_eff (HCO₃⁻) | ~5 mM | CCM-corrected estimate | DEFAULT |
| CCM concentration factor | ~1000× | Cyanobacteria literature | DEFAULT |
| Zarrouk medium DIC | 200 mM | NaHCO₃ 16.8 g/L | KNOWN |
| K1, K2 (carbonate) | Tabulated f(T) | Physical chemistry | KNOWN |

### To be determined (Spirulina-specific)

| Constant | Expected range | Method |
|---|---|---|
| Ea_repair (Spirulina) | 50–90 kJ/mol | PI curves at 3+ temperatures |
| Tm (Spirulina) | 40–45°C | Repair assay above 38°C |
| CCM efficiency vs pH | Unknown | Literature or measurement |

---

## 11. Dynamic Simulation Results

PBR feedback simulation (pbr_dynamics.py) confirmed:

| Scenario | T | CO₂ control | Max biomass | Bottleneck |
|---|---|---|---|---|
| S1 | 25°C | None | 0.9 g/L | β_thermal |
| S2 | 25°C | Yes | 0.9 g/L | β_thermal (CO₂ helps but T still limits) |
| S3 | 35°C | Yes | 7.7 g/L | β_carbon (fast growth depletes DIC) |
| S4 | 35°C | Yes + N | 9.6 g/L | β_carbon eventually |

Key findings:
1. At 25°C, CO₂ control alone cannot rescue growth — temperature is the primary bottleneck
2. At 35°C, growth is fast enough that CO₂ supply becomes limiting — the bottleneck switches
3. Nitrogen becomes limiting only after both T and CO₂ are controlled
4. pH runaway occurs when Pgross × biomass exceeds CO₂ supply capacity

---

## 12. Validation Summary

| Prediction | Data source | Result |
|---|---|---|
| kd is T-independent | Allakhverdiev 2004 (4 temps) | CV = 0.9% ✓ |
| kr is Arrhenius-type | Allakhverdiev 2004, Ueno 2016 | Ea = 48–87 kJ/mol ✓ |
| α is T-independent | Rehder 2023 (diatom, 6°C vs 15°C) | Ratio = 1.01 ✓ |
| Denaturation at ~42°C | Ueno 2016 (repair abolished at 44°C) | ✓ |
| pH 12 kills growth | Kobayashi 1996 (Spirulina NIES-39) | ✓ |
| CO₂ control → 2× biomass | Kobayashi 1996 (1.2 → 2.2 g/L) | ✓ |
| pH + N → 4× biomass | Kobayashi 1996 (1.2 → 4.2 g/L) | ✓ |
| Acclimation ≠ acute T response | Rehder 2023 (×1.22 vs ×1.90) | ✓ |
| ROS inhibits repair not damage | Nishiyama 2001 (translation block) | ✓ |
| ATP is rate-limiting for repair | Allakhverdiev 2005 (DCCD abolishes) | ✓ |

---

## 13. Data Repository

```
miodori-no-tamaki-main/
├── ARCHITECTURE_unified.md          ← THIS FILE
├── ARCHITECTURE_thermal.md          ← Phase 1 design (historical)
├── ARCHITECTURE_carbon.md           ← Phase 2 design (historical)
├── eos_sensor.py                    ← Original EOS implementation
├── data/
│   ├── SOURCES.md                   ← Literature catalog
│   ├── torzillo1994_*.csv (5)       ← PI curve temperature data
│   ├── torzillo1991_*.csv (6)       ← PBR night biomass loss
│   ├── torzillo1998_*.csv (3)       ← PAM fluorescence data
│   ├── tanaka2020_*.csv (5)         ← Dark-period temperature effects
│   ├── kobayashi1996_*.csv (2)      ← pH control growth data
│   ├── nishiyama_group_*.csv (2)    ← Independent Ea_r validation
│   ├── rehder2023_*.csv (1)         ← Cross-species validation
│   └── arrhenius_fitted_params.csv  ← Fitted Arrhenius parameters
├── soba/
│   ├── fit_arrhenius_kd_kr.py       ← Arrhenius fitting script
│   └── pbr_dynamics.py              ← PBR feedback simulation
├── figures/
│   ├── Fig_arrhenius_kd_kr.png
│   ├── Fig_SAI_crossover_prediction.png
│   ├── Fig_cross_species_validation.png
│   └── Fig_PBR_dynamics_simulation.png
└── raw_data/
    └── csv.zip                      ← piCurve 1,808 curves
```

---

## 14. Roadmap

### Phase 1: Spirulina EOS-thermal (THIS SESSION — largely complete)
- [x] β_eff = min(β_light, β_thermal, β_carbon) formulation
- [x] Arrhenius fitting (Ea_r initial + correction)
- [x] Independent validation (Nishiyama group, 4 papers)
- [x] Cross-species validation (Rehder 2023, diatom)
- [x] SAI decomposition theory
- [x] PBR feedback simulation
- [x] pH-CO₂ channel formalization
- [ ] Spirulina-specific Ea_r (piCurve metadata or collaboration)

### Phase 2: Implementation
- [ ] eos_thermal.py module (closed-form functions)
- [ ] Operator dashboard prototype
- [ ] Validation against Kobayashi 1996 growth curves (quantitative)

### Phase 3: Generalization
- [ ] Higher plants (lettuce, strawberry)
- [ ] Allocation layer η(T, N) for target-specific optimization
- [ ] Amino acid composition control (protein quality)

### Phase 4: Product
- [ ] Miosync PBR monitoring system (pH + T + EOS)
- [ ] Layer 3 optimization (ML or model-based)
- [ ] Field validation with partner facilities

---

*"The EOS transforms PI curves from fitted objects into predictable state functions.
The thermal extension transforms environmental stress from unmeasured noise into diagnosed bottlenecks."*

— M. Iizumi & T. Iizumi, Miosync Inc.
