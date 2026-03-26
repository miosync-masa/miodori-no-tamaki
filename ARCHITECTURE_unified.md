# Midori-no-Tamaki: Unified Architecture (Paper v2)
## EOS Environmental Bottleneck Diagnosis for Photosynthetic Systems

**Authors:** M. Iizumi & T. Iizumi (Miosync, Inc.)
**Date:** 2025-03-22 (updated to paper v2 notation)
**Paper:** "Closed-Form Environmental Bottleneck Diagnosis for Photosynthetic Systems"
**Target:** Biotechnology & Bioengineering
**Status:** Theory established, cross-species validated, paper draft complete

---

## 1. Design Philosophy

The original EOS (Part I) transforms PI curves from fitted objects into predictable state functions using (α, Pmax). This extension applies the same philosophy to environmental stress: **predict the photoinhibition state from minimal, real-time sensor readings using closed-form physics**.

Three design commitments:

1. **Closed-form only.** No differential equations, no iterative solvers, no trained models in the state-estimation layer.
2. **Species-agnostic structure.** Functional forms are general; species-specific differences enter through parameter values.
3. **Separation of diagnosis from control.** "What is limiting now?" is answered; "What should be done?" is a separate layer.

---

## 2. Core Formulation (Paper v2 — Capacity Factor Approach)

### 2.1 The Key Insight

Environmental stresses do NOT accelerate photodamage (kd). They inhibit REPAIR (kr).
Therefore: β increases when repair capacity drops.

### 2.2 Capacity Factors

```
b_thermal(T) = kr(T) / kr(T_opt)  ∈ (0, 1]    — repair capacity relative to optimum
b_carbon(pH, DIC, T) = Ci_eff / (Km_eff + Ci_eff)  ∈ (0, 1]  — carbon sufficiency
```

Both are dimensionless, with 1 = optimal and 0 = severe stress.

### 2.3 Bottleneck-Minimum (Liebig's Law)

```
b_env = min(b_thermal, b_carbon)
```

The most constrained channel determines the overall environmental capacity.

### 2.4 Effective Susceptibility

```
β_eff = β_ref(α) / b_env
```

- β_ref(α) = baseline susceptibility from Part I scaling law (α–β relationship)
- When b_env < 1: β_eff > β_ref → culture is MORE susceptible to photoinhibition
- When b_env = 1: β_eff = β_ref → original EOS (no environmental stress)

**Critical:** β_eff is a function of (α, T, pH, DIC) but NOT of irradiance I.
β is a curve-level parameter, not a function of instantaneous light.

### 2.5 SAI Decomposition (min–max Duality)

```
SAI = log₁₀(β_eff / β_ref) = −log₁₀(b_env) = max(SAI_thermal, SAI_carbon)

SAI_thermal(T) = −log₁₀(b_thermal) ≈ Ea_r/(2.303·Rg) × (1/T − 1/T_opt)   [cold branch]
SAI_carbon(pH, DIC, T) = −log₁₀(b_carbon)
```

The min on capacity maps to max on SAI. Dominant bottleneck = highest SAI component.

---

## 3. Channel 1: b_thermal — Temperature-Dependent Repair

### 3.1 Physics

D1 repair = FtsH protease (degradation) → psbA translation → D1 synthesis → PSII reassembly.
All steps are enzymatic, ATP-dependent, and temperature-sensitive.

### 3.2 Closed Form

```
kr(T) = kr0 · exp(−Ea_r / Rg·T) · 1/(1 + exp((T − Tm)/δT))
        |---- Arrhenius cold branch ----|  |-- denaturation sigmoid --|

b_thermal(T) = kr(T) / kr(T_opt)

Explicit:
b_thermal(T) = exp(−Ea_r/Rg · (1/T − 1/T_opt)) · [1+exp((T_opt−Tm)/δT)] / [1+exp((T−Tm)/δT)]
```

### 3.3 Parameters

| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| Ea_r | 63 kJ/mol (0.65 eV) | 48–87 kJ/mol | Literature + cross-species validation |
| T_opt | Species-dependent | 30–37°C | Species-specific |
| Tm | T_opt + 7°C (heuristic) | 40–45°C | Ueno 2016 (Synechocystis: 42–44°C) |
| δT | 3°C | 2–5°C | Estimated from Ueno 2016 |

### 3.4 Key Properties (Validated)

- kd is temperature-INDEPENDENT (CV = 0.9%, 10–34°C) — Allakhverdiev 2004
- kr follows Arrhenius (Ea = 48–87 kJ/mol across species) — Allakhverdiev 2004, Rehder 2023
- α is temperature-INDEPENDENT (ratio = 1.01) — Rehder 2023
- Denaturation threshold exists (repair abolished at 44°C) — Ueno 2016
- Acute response ≠ acclimated response (×1.90 vs ×1.22) — Rehder 2023

### 3.5 Operational Significance

```
I_cross(T) = kr(T) / σ_PSII   — irradiance where damage = repair

Low T → I_cross drops → photoinhibition at lower light levels
→ Morning cold = photoinhibition risk even at moderate light
```

---

## 4. Channel 2: b_carbon — Carbon Supply Limitation

### 4.1 Physics

Calvin cycle substrate limitation → electron accumulation on PSI → ROS → EF-G oxidation
→ D1 translation block → kr inhibited indirectly.

### 4.2 Carbonate Equilibrium (closed form)

```
DIC = [CO₂] + [HCO₃⁻] + [CO₃²⁻]

α₁(pH, T) = [HCO₃⁻]/DIC = 1 / ([H⁺]/K₁ + 1 + K₂/[H⁺])

Ci_eff = DIC × α₁(pH, T)     — effective carbon available to CCM

K₁(T) ≈ 4.3×10⁻⁷ at 25°C  (tabulated)
K₂(T) ≈ 4.7×10⁻¹¹ at 25°C  (tabulated)
```

For cyanobacteria: HCO₃⁻ is the dominant CCM-accessible species.
More generally: Ci_eff = effective inorganic carbon pool accessible to organism-specific CCM.

### 4.3 Carbon Capacity Factor

```
b_carbon(pH, DIC, T) = Ci_eff / (Km_eff + Ci_eff)    — Michaelis-Menten

Ci_eff ≫ Km_eff → b_carbon → 1 (carbon-replete)
Ci_eff → 0      → b_carbon → 0 (growth stops)
```

### 4.4 Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| Zarrouk DIC | 200 mM | NaHCO₃ 16.8 g/L |
| Km_eff | ~5 mM | CCM-corrected estimate |
| K₁, K₂ | f(T), tabulated | Physical chemistry |

### 4.5 Key Properties (Validated)

- pH 12 → HCO₃⁻ < 10% of DIC → growth stops — Kobayashi 1996
- Sequential bottleneck transitions: carbon → nitrogen → light — Kobayashi 1996
- 3.5× biomass increase upon serial constraint relief — Kobayashi 1996

---

## 5. Feedback Loops and Dynamic Behavior

### 5.1 The pH–CO₂ Feedback

```
High light → fast Pgross → CO₂ consumed → pH rises →
→ HCO₃⁻ decreases → b_carbon drops → Pgross limited → pH stabilizes
  (NEGATIVE FEEDBACK within pH 8.5–10.5)

BUT: pH > ~11 → HCO₃⁻ → CO₃²⁻ transition accelerates →
→ RUNAWAY to pH 12+ → irreversible growth stop
```

### 5.2 Temperature Trade-off

Temperature increase simultaneously:
1. IMPROVES b_thermal (kr↑, Arrhenius)
2. WORSENS CO₂ solubility (Henry's law)
3. INCREASES respiration R(T) → nighttime biomass loss
4. INCREASES CO₂ consumption rate → faster pH rise

→ Higher T requires more aggressive CO₂ supply.

### 5.3 Interaction Matrix

```
              Light(I)    Temp(T)     CO₂/pH      Nitrogen(N)
b_thermal     indirect    DIRECT      indirect    —
b_carbon      indirect    indirect    DIRECT      —
```

Each channel has ONE primary control knob. Cross-talk occurs through feedback but does not change the primary control structure.

---

## 6. Sensing Architecture

### 6.1 Minimum Sensor Set

| Sensor | Measures | Feeds | Cost |
|--------|----------|-------|------|
| Thermistor | T(t) | b_thermal | ~$10 |
| pH probe | pH(t) | b_carbon | ~$10 |
| (Medium recipe) | DIC₀ | b_carbon | Free (nominal, optionally recalibrated) |
| PAM (optional) | α, Pmax | β_ref, S regime | ~$10k+ |

**Minimum viable system: pH + Temperature + known DIC.**

### 6.2 Computation (all closed-form, instant)

```python
# At each sensor update:
Ci_eff = DIC * alpha1(pH, T)                               # carbonate equilibrium
b_carbon = Ci_eff / (Km_eff + Ci_eff)                      # Michaelis-Menten
b_thermal = exp(-Ea_r/Rg*(1/T_K - 1/T_opt_K)) * sigmoid    # Arrhenius + denaturation
b_env = min(b_thermal, b_carbon)                            # bottleneck
# β_eff = β_ref(α) / b_env                                 # if PAM available
# SAI = -log10(b_env)                                      # stress index
```

Computation is negligible relative to sensor update intervals. Compatible with low-cost embedded implementation.

### 6.3 Operator Decision Logic

| Condition | Diagnosis | Recommended action |
|-----------|-----------|-------------------|
| b_carbon < b_thermal | Carbon-limited | Increase CO₂; verify pH 8.5–10.5 |
| b_thermal < b_carbon | Temperature-limited | Warm culture toward T_opt |
| Both > 0.8, plateau | Outside formalized channels | Check N; consider harvesting |
| pH > 10.5 | Approaching carbon crisis | Immediate CO₂ injection |
| T > Tm − 5°C | Approaching denaturation | Cool culture |

Thresholds are illustrative heuristics; tune to organism and operating context.

### 6.4 Layer Structure

```
Layer 0: Sensors       → pH, T, (PAM optional)        [hardware]
Layer 1: State est.    → b_env, bottleneck ID          [closed-form, instant]
Layer 2: Threshold     → Alarms, operator guidance     [rule-based]
Layer 3: Optimization  → Objective-dependent control   [ML/model-based, FUTURE]
```

Layers 0–2: implementable today. Layer 3: future work.
The closed-form state estimator provides physics-informed features for any Layer 3 method.

---

## 7. Cross-Species Validation Summary

| # | Prediction | Data | Species | Type | Result |
|---|------------|------|---------|------|--------|
| 1 | kd T-independent | CV = 0.9% (10–34°C) | Synechocystis | Direct | Confirmed |
| 2a | kr Arrhenius | Ea = 74–87 kJ/mol | Synechocystis | Direct | Confirmed |
| 2b | kr Arrhenius | Ea = 48 kJ/mol (acute Pmax) | Phaeodactylum | Indirect | Consistent |
| 3 | Denaturation ceiling | Repair abolished at 44°C | Synechocystis | Direct | Confirmed |
| 4 | α T-independent | Ratio = 1.01 (6°C vs 15°C) | Phaeodactylum | Direct | Confirmed |
| 5 | b_carbon → 0 at pH 12 | Growth stops, culture bleaches | Spirulina | Qualitative | Consistent |
| 6 | Sequential transitions | C → N → light upon serial relief | Spirulina | Qualitative | Consistent |
| 7 | SAI decomposable | Acute ≠ acclimated (×1.90 vs ×1.22) | Phaeodactylum | Indirect | Consistent |

7/7 predictions supported. Structure is species-agnostic; parameters are species-dependent.

---

## 8. Physical Constants

### Confirmed

| Constant | Value | Source |
|----------|-------|--------|
| kd T-independence | CV = 0.9% | Allakhverdiev 2004 |
| Ea_repair (default) | 63 kJ/mol (0.65 eV) | Literature consensus |
| Ea_repair range | 48–87 kJ/mol | Cross-species (3 studies) |
| Denaturation Tm | 42–44°C (Synechocystis) | Ueno 2016 |
| α T-independence | Ratio = 1.01 | Rehder 2023 |
| γ₀ | cosh²(1) ≈ 2.381 | Original EOS |
| α–β scaling (m, c) | 0.814, −1.355 | Original EOS |

### Adopted Defaults

| Constant | Value | Basis |
|----------|-------|-------|
| Km_eff (HCO₃⁻) | ~5 mM | CCM-corrected estimate |
| Zarrouk DIC | 200 mM | NaHCO₃ 16.8 g/L |
| Tm heuristic | T_opt + 7°C | Pragmatic starting point |
| δT | 3°C | Estimated |

---

## 9. Repository Structure

```
miodori-no-tamaki-main/
├── ARCHITECTURE_unified.md          ← THIS FILE (paper v2 notation)
├── eos_sensor.py                    ← Original EOS implementation
├── data/
│   ├── SOURCES.md
│   ├── torzillo1994_*.csv (5)       ← PI curve temperature data
│   ├── torzillo1991_*.csv (6)       ← Night biomass loss
│   ├── torzillo1998_*.csv (3)       ← PAM fluorescence
│   ├── tanaka2020_*.csv (5)         ← Dark-period temperature
│   ├── kobayashi1996_*.csv (2)      ← pH control growth
│   ├── nishiyama_group_*.csv (2)    ← Independent Ea validation
│   ├── rehder2023_*.csv (1)         ← Cross-species validation
│   └── arrhenius_fitted_params.csv
├── soba2/
│   ├── fit_arrhenius_kd_kr.py       ← Arrhenius fitting (Fig 1-2)
│   ├── generate_thermal_figures.py  ← Cross-species + pH-CO₂ (Fig 3-4)
│   └── pbr_dynamics.py             ← PBR simulation (Fig 5)
└── figures2/
    ├── Fig_arrhenius_kd_kr.png
    ├── Fig_SAI_crossover_prediction.png
    ├── Fig_cross_species_validation.png
    ├── Fig_pH_CO2_interaction_map.png
    └── Fig_PBR_dynamics_simulation.png
```

---

## 10. Roadmap

### Done
- [x] β_eff = β_ref(α) / b_env formulation (paper v2)
- [x] b_thermal: Arrhenius + denaturation sigmoid
- [x] b_carbon: carbonate equilibrium + Michaelis-Menten
- [x] SAI = −log₁₀(b_env) = max(SAI_T, SAI_C) — min–max duality
- [x] Cross-species validation (7 predictions, 3 species)
- [x] PBR feedback simulation (4 scenarios)
- [x] Paper draft complete (Abstract–References, 6,700+ words)
- [x] Code notation unified to paper v2

### Next
- [ ] Spirulina-specific Ea_r calibration
- [ ] Fig 1 conceptual diagram (β_ref → β_eff)
- [ ] Final formatting for B&B submission
- [ ] eos_thermal.py implementation module

### Future
- [ ] Higher plants extension (strawberry, lettuce)
- [ ] Allocation layer η(T, N)
- [ ] Nitrogen channel formalization
- [ ] Layer 3 optimal control
- [ ] Nighttime respiration integration

---

*"The EOS transforms PI curves from fitted objects into predictable state functions.*
*The thermal/carbon extension transforms environmental stress from unmeasured noise into diagnosed bottlenecks.*
*b_env = min(b_thermal, b_carbon). β_eff = β_ref / b_env. SAI = max(SAI_T, SAI_C)."*

— M. Iizumi & T. Iizumi, Miosync Inc.
