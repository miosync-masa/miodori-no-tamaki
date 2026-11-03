# 🌿 Midori-no-Tamaki（緑の環）

**Equation of State for Photosynthesis–Irradiance Curves**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

> *Predict a complete photoinhibition curve — including optimal irradiance — from just two sensor readings.*

---

## What is this?

Photosynthesis–irradiance (PI) curves are central to aquatic ecology, algal biotechnology, and photobioreactor control. The standard workflow is: **measure many light–response points → fit a 4-parameter model**. This is slow, expensive, and incompatible with real-time monitoring.

**Midori-no-Tamaki** introduces an **Equation of State (EOS)** that collapses the PI parameter space:

| Mode | Inputs | What you get |
|------|--------|-------------|
| **EOS2** | α, P_max | Full PI curve + regime + I_opt |
| **EOS3** | α, P_max, SAI | Full PI curve with stress correction |

The key insight: the photoinhibition parameter β is **not independent** of the light-harvesting parameter α. A scaling law (log β = 0.814 · log α − 1.355, r² = 0.43, N = 1808) enables parameter prediction, and a single **Stress Adaptation Index (SAI)** captures the residual biological variation.

### ✨ NEW: Closed-Form Optimal Irradiance

The optimal irradiance — the light level that maximizes photosynthesis — is a **weighted geometric mean** of the saturation and inhibition irradiances:

```
I_opt = I_α^(1/(γ₀+1)) · I_β^(γ₀/(γ₀+1))
```

where I_α = P_max/α (saturation), I_β = P_max/β (inhibition), and the weights are determined solely by γ₀ = cosh²(1). No fitting required. P(I_opt) within **0.2%** of the numerical optimum.

**Physical meaning:** I_opt sits between saturation and inhibition, weighted 70.4% toward the inhibition scale. The gate shape γ₀ — the same constant that fixes the SCC profile and makes the design law approximately regime-invariant (k_R1 = 51.6, k_R2 = 48.8, ratio 1.06) — also determines the optimal point.

## Quick Start

### Installation

```bash
git clone https://github.com/miosync-masa/miodori-no-tamaki.git
cd miodori-no-tamaki
# No dependencies required for core module (pure Python, stdlib only)
```

### Predict a PI curve (2 lines of Python)

```python
from eos_sensor import EOSSensor

sensor = EOSSensor()
result = sensor.predict(alpha=0.05, Pmax=8.0)

print(result.regime)              # "R1"
print(result.eos_tier)            # "EOS2"
print(result.I_opt)               # Optimal irradiance (numerical peak)
print(result.I_opt_closed_form)   # Optimal irradiance (analytic formula)
print(result.curve[:3])           # PI curve as list of dicts
```

### Closed-form I_opt directly

```python
from eos_sensor import I_opt_analytic, beta_pred, W_ALPHA, W_BETA

# From α and Pmax alone (EOS2 mode)
alpha, Pmax = 0.05, 8.0
beta = beta_pred(alpha)
I_opt = I_opt_analytic(alpha, Pmax, beta)
print(f"Optimal irradiance: {I_opt:.0f} µmol m⁻² s⁻¹")

# Validate against numerical search
from eos_sensor import validate_I_opt
result = validate_I_opt(alpha, Pmax, beta)
print(f"P error: {result['P_error_pct']:.4f}%")
```

### Command line

```bash
# Basic prediction (human-readable output)
python eos_sensor.py --alpha 0.05 --Pmax 8.0

# With stress index and dark offset
python eos_sensor.py --alpha 0.03 --Pmax 6.0 --SAI 0.15 --R 0.5

# JSON output (for piping to other tools)
python eos_sensor.py --alpha 0.05 --Pmax 8.0 --json --compact

# Diagnose an existing PI fit
python eos_sensor.py --alpha 0.05 --Pmax 8.0 --beta-obs 0.01

# Instrument design specification
python eos_sensor.py --alpha 0.05 --Pmax 8.0 --target-NRMSE 5.0

# Start REST API server
python eos_sensor.py --serve --port 5050
```

### REST API

```bash
python eos_sensor.py --serve

# Predict
curl -X POST http://localhost:5050/predict \
  -H "Content-Type: application/json" \
  -d '{"alpha": 0.05, "Pmax": 8.0}'

# Diagnose
curl -X POST http://localhost:5050/diagnose \
  -H "Content-Type: application/json" \
  -d '{"alpha": 0.05, "Pmax": 8.0, "beta_obs": 0.01}'
```

Flask is required only for the API server (`pip install flask`). The core module has **zero dependencies**.

## The Physics

### PCC × SCC Model (Ph10)

```
P(I) = Pmax · PCC(I) · SCC(I) − R

PCC(I) = tanh(αI / Pmax)           ← light harvesting (saturates)
SCC(I) = tanh((Pmax / βI)^γ₀)      ← stress coupling  (activates at high I)
γ₀     = cosh²(1) ≈ 2.381          ← universal gate shape
```

Two channels, one curve. PCC captures how photosynthesis saturates; SCC captures how it breaks down under excess light. Their product is the full PI response.

### Gate Variable S

```
S = α / β = I_β / I_α
```

S measures how well-separated the two channels are:

| Regime | S range | Population | EOS accuracy |
|--------|---------|-----------|-------------|
| **R1** (factorized) | S > 10 | 67.5% | median R² = 0.929 |
| **R2** (transition) | 3 < S ≤ 10 | 31.6% | R² = 0.935 (with SAI) |
| **R3** (coupled) | S ≤ 3 | 0.9% | EOS not valid |

### The α–β Scaling Law

```
log₁₀β = 0.814 · log₁₀α − 1.355
```

This is the engine of the EOS. Across 1,808 PI curves spanning diverse marine phytoplankton, β is **constrained** by α. Not perfectly (r² = 0.43), but enough to predict curves. The residual is SAI.

### Optimal Irradiance (Closed-Form)

```
I_opt = I_α^w_α · I_β^w_β

where w_α = 1/(γ₀+1) = 0.296,  w_β = γ₀/(γ₀+1) = 0.704
```

Derived from dP/dI = 0 in the double-tanh architecture. At the optimum, PCC and SCC gate arguments balance, yielding a power-law in S:

```
I_opt / I_α = S^(γ₀/(γ₀+1))
```

This is a **weighted geometric mean** of the saturation irradiance (I_α = P_max/α) and the inhibition irradiance (I_β = P_max/β). The weights are determined by γ₀ alone.

| Property | Value |
|----------|-------|
| Weight on I_β (inhibition) | γ₀/(γ₀+1) = 0.704 |
| Weight on I_α (saturation) | 1/(γ₀+1) = 0.296 |
| P(I_opt) accuracy | < 0.2% of numerical optimum (S > 3) |
| Valid regimes | R1 and R2 |
| Note | In R1 (S > 10), plateau is broad; I_opt location is irrelevant |

### Three Convergences of γ₀

The canonical gate shape γ₀ = cosh²(1) appears in three independent results:

1. **Gate shape invariance**: Ph10 (γ fixed) vs Ph11 (γ free) show ΔR²_adj < 0.001 for 73.8% of curves
2. **Approximate regime-invariance of the design law**: NRMSE = 50.6·σ_SAI holds across R1 and R2 to within 6% (k_R1 = 51.6, k_R2 = 48.8; seed-to-seed SD of k 1.3, i.e. 2.6%) because γ₀ fixes the gate profile
3. **I_opt weights**: The optimal irradiance weights w_α = 1/(γ₀+1) and w_β = γ₀/(γ₀+1) are determined by γ₀

All three emerge from the same constant. The gate shape determines the profile, the error sensitivity, and the optimal point.

### Stress Adaptation Index (SAI)

```
SAI = log₁₀(β_obs) − log₁₀(β_pred(α))
```

SAI is a single number that captures everything the scaling law misses: acclimation state, species composition, nutrient status. Positive SAI = stressed; negative SAI = photoprotected. In a PBR, a rising SAI is an early warning signal.

### Design Law

```
NRMSE(%) = 50.6 × σ_SAI
```

This linear relationship (R² ≈ 0.999; k averaged over 100 noise realisations, seed-to-seed SD 1.3) directly translates sensor precision (σ\_SAI) into prediction accuracy (NRMSE). Want 5% accuracy? You need σ\_SAI < 0.10.

## Environmental Bottleneck Extension

The EOS framework extends to environmental stress diagnosis. Environmental factors (temperature, carbon supply) modify photoinhibition not by accelerating damage, but by inhibiting repair (Nishiyama & Murata, 2014).

### Capacity Factors

```
b_thermal(T)           = k_r(T) / k_r(T_opt)            ∈ (0, 1]
b_carbon(pH, DIC, T)   = C_i_eff / (K_m_eff + C_i_eff)  ∈ (0, 1]
b_env                  = min(b_thermal, b_carbon)

β_eff = β_ref(α) / b_env
SAI   = −log₁₀(b_env) = max(SAI_thermal, SAI_carbon)
```

When environmental capacity drops (b_env < 1), photoinhibition susceptibility increases. The **minimum capacity** identifies the dominant bottleneck — and which knob to turn.

### Minimum Sensors

| Sensor | Cost | Feeds |
|--------|------|-------|
| Temperature | ~$10 | b_thermal |
| pH probe | ~$10 | b_carbon |
| Medium recipe | Free | DIC (nominal) |
| PAM (optional) | ~$10k | α, Pmax for full EOS |

See: Iizumi & Iizumi (2026), "Closed-Form Environmental Bottleneck Diagnosis for Photosynthetic Systems" (in preparation for Biotechnology & Bioengineering).

## Output Structure

```python
result = sensor.predict(alpha=0.05, Pmax=8.0, SAI=0.15)
```

```json
{
  "alpha": 0.05,
  "Pmax": 8.0,
  "SAI": 0.15,
  "R": 0.0,
  "beta_predicted": 0.003854,
  "beta_effective": 0.005447,
  "S": 9.18,
  "regime": "R2",
  "regime_label": "Transition — SCC affects plateau",
  "eos_tier": "EOS3",
  "in_forbidden_zone": false,
  "I_alpha": 160.0,
  "I_beta": 1468.9,
  "I_opt": 845.0,
  "I_opt_closed_form": 869.3,
  "curve": [...]
}
```

## Forbidden Zone

The **forbidden zone** is the window centred on the gate-overlap point S_c = 1 (I_α = I_β), |log₁₀ S| ≤ 0.1465, i.e. 0.714 ≤ S ≤ 1.401. The centre follows from the PCC–SCC architecture; the half-width is half the log-width of the largest empirical gap in the S < 3 band (0.82–1.61). Only 3 of 1808 curves fall in it.

The primary evidence is an α–β permutation test: shuffling α and β independently (marginals preserved, coupling destroyed) raises the expected occupancy to 30.0 ± 4.8 curves (Z = −5.6, p < 10⁻⁴). Re-fitting synthetic curves from noisy RLC data does push some out of the zone (32% retained), but this accounts for only part of the depletion (retention-adjusted null 9.5 ± 3.0 vs 3 observed, P = 0.012). See `analysis/output/canonical_tables.md`.

## Repository Structure

```
midori-no-tamaki/
├── README.md                    ← you are here
├── LICENSE                      ← MIT
├── eos_sensor.py                ← soft sensor module (zero dependencies)
├── ARCHITECTURE_unified.md      ← thermal extension design document
├── raw_data/
│   ├── csv.zip                  ← piCurve fits of 1,808 PI curves (Opt_ParVal_of_piModels.csv, ph10_*.csv, ...)
│   └── data/                    ← literature tables for the thermal/carbon extension
│       ├── SOURCES.md
│       ├── torzillo1994_*.csv   ← temperature response data
│       ├── kobayashi1996_*.csv  ← pH control growth data
│       ├── rehder2023_*.csv     ← cross-species validation
│       └── ...
├── analysis/                    ← canonical rerun of Part I numbers (seed-fixed)
│   ├── ph10_fit.py              ← Ph10 model + multi-start LM fit with R² filter
│   ├── rerun_canonical.py       ← end-to-end rerun → output/canonical.json, canonical_tables.md
│   └── output/
├── figures/                     ← Part I main and SI figures and their generators
│   ├── generate_fig1_framework.py ← Fig 1
│   ├── generate_main_figures.py ← Fig 3, 4, 5
│   ├── generate_fig2_phase_diagram.py ← Fig 2
│   ├── generate_fig9_workflow.py ← Fig 9
│   ├── generate_si_figures.py   ← Fig S1–S3, Table S2
│   └── Fig*.png, Table_S2_extended_stats.csv  (file names follow manuscript numbering)
└── soba2/                       ← EOS thermal/carbon extension scripts
    ├── eos_thermal_model.py         ← shared closed-form equations (Eqs. 6-12)
    ├── fit_allakhverdiev_kd_kr.py   ← Fig 6 (Arrhenius, I_cross, b_thermal)
    ├── generate_thermal_figures.py  ← Fig 7 (pH–CO2) and Fig S4 (cross-species)
    ├── pbr_dynamics.py              ← Fig 8 (PBR simulation)
    └── output/                      ← generated figures (Fig 6–8, S4) and JSON summaries
```

## Data Source

All analyses are based on **1,808 PI curves** from the piCurve compilation:

> Amirian, M.M., Finkel, Z.V., Devred, E. & Irwin, A.J. (2025). Parameterization of photoinhibition for phytoplankton. *Communications Earth & Environment* 6:707.
> Dataset: Amirian, M.M., Finkel, Z.V., Devred, E. & Irwin, A.J. (2025). Parameterization of Photoinhibition for Phytoplankton [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16748102

## Citation

```bibtex
@article{iizumi2026eos,
  title   = {An equation of state for photoinhibition: a forbidden zone,
             a diagnosable stress index, and environmental bottleneck channels},
  author  = {Iizumi, Masamichi},
  journal = {Photosynthesis Research},
  year    = {2026},
  note    = {Submitted}
}
```

## Authors

**Masamichi Iizumi** — Miosync, Inc.

---

*"The measure-then-fit era is over. The EOS lets the physics do the work."*

🌿 **Midori-no-Tamaki** — where photosynthesis meets thermodynamics.
