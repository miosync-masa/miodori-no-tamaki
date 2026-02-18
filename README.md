# 🌿 Midori-no-Tamaki（緑の環）

**Equation of State for Photosynthesis–Irradiance Curves**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

> *Predict a complete photoinhibition curve from just two sensor readings.*

---

## What is this?

Photosynthesis–irradiance (PI) curves are central to aquatic ecology, algal biotechnology, and photobioreactor control. The standard workflow is: **measure many light–response points → fit a 4-parameter model**. This is slow, expensive, and incompatible with real-time monitoring.

**Midori-no-Tamaki** introduces an **Equation of State (EOS)** that collapses the PI parameter space:

| Mode | Inputs | What you get |
|------|--------|-------------|
| **EOS2** | α, P_max | Full PI curve + regime classification |
| **EOS3** | α, P_max, SAI | Full PI curve with stress correction |

The key insight: the photoinhibition parameter β is **not independent** of the light-harvesting parameter α. A universal scaling law (log β = 0.814 · log α − 1.355, r² = 0.43, N = 1808) enables parameter prediction, and a single **Stress Adaptation Index (SAI)** captures the residual biological variation.

## Quick Start

### Installation

```bash
git clone https://github.com/miosync-inc/midori-no-tamaki.git
cd midori-no-tamaki
# No dependencies required for core module (pure Python, stdlib only)
```

### Predict a PI curve (2 lines of Python)

```python
from eos_sensor import EOSSensor

sensor = EOSSensor()
result = sensor.predict(alpha=0.05, Pmax=8.0)

print(result.regime)        # "R1"
print(result.eos_tier)      # "EOS2"
print(result.I_opt)         # Optimum irradiance (µmol m⁻² s⁻¹)
print(result.curve[:3])     # PI curve as list of dicts
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

# Design spec
curl -X POST http://localhost:5050/design \
  -H "Content-Type: application/json" \
  -d '{"target_NRMSE_pct": 5.0}'

# Health check
curl http://localhost:5050/health
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

### Stress Adaptation Index (SAI)

```
SAI = log₁₀(β_obs) − log₁₀(β_pred(α))
```

SAI is a single number that captures everything the scaling law misses: acclimation state, species composition, nutrient status. Positive SAI = stressed; negative SAI = photoprotected. In a PBR, a rising SAI is an early warning signal.

### Design Law

```
NRMSE(%) = 50.4 × σ_SAI
```

This linear relationship (R² ≈ 0.999) directly translates sensor precision (σ\_SAI) into prediction accuracy (NRMSE). Want 5% accuracy? You need σ\_SAI < 0.10.

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
  "expected_NRMSE_pct": null,
  "sigma_SAI": null,
  "I_alpha": 160.0,
  "I_beta": 1468.9,
  "I_opt": 845.0,
  "curve": [
    {"I": 1.0, "P_gross": 0.05, "P_net": 0.05, "PCC": 0.006, "SCC": 1.0},
    ...
  ]
}
```

## Forbidden Zone

Within the low-S region, a **forbidden zone** (0.82 < S < 1.61) exists where almost no real phytoplankton populations are found. This structural gap in the α–β phase space is statistically robust (p < 0.001 by split-half consistency tests) and may reflect a fundamental biophysical constraint on the PCC–SCC coupling.

## Repository Structure

```
midori-no-tamaki/
├── README.md               ← you are here
├── LICENSE                  ← MIT
├── eos_sensor.py            ← soft sensor module (zero dependencies)
├── FACTSHEET.md             ← detailed technical reference
├── paper/
│   └── draft_v6_bej.md      ← manuscript (BEJ submission)
├── raw_data/
│   ├── ph10_with_SAI.csv    ← 1,808 PI curves with SAI
│   ├── ph10_extended.csv    ← extended parameter table
│   ├── fig*.png             ← publication figures
│   └── ...
└── soba/                    ← analysis scripts ("蕎麦" = buckwheat noodles)
    ├── p5_forbidden_zone_validation.py
    ├── outlier_catalog.py
    ├── pi_final_figure.py
    └── ...
```

## Data Source

All analyses are based on **1,808 PI curves** from the piCurve compilation:

> Amirian, M.A. et al. (2025). Parameterization of photoinhibition for phytoplankton. *Communications Earth & Environment* 6:707.
> Dataset: Amirian, M.A. & Irwin, A.J. (2025). piCurve R package. Zenodo. https://doi.org/10.5281/zenodo.16748102

## Citation

If you use this code or the EOS framework:

```bibtex
@article{iizumi2026eos,
  title   = {A sensing-ready equation of state for photoinhibition:
             predicting {PI} curves from $\alpha$, $P_{\max}$,
             and a stress adaptation index},
  author  = {Iizumi, Masamichi and Iizumi, Tamaki},
  journal = {Biochemical Engineering Journal},
  year    = {2026},
  note    = {Submitted}
}
```

## Authors

**Masamichi Iizumi** —  Miosync, Inc.
**Tamaki Iizumi** —  Miosync, Inc.

---

*"The measure-then-fit era is over. The EOS lets the physics do the work."*

🌿 **Midori-no-Tamaki** — where photosynthesis meets thermodynamics.
