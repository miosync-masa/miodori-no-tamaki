# Data Sources for Spirulina EOS Thermal Extension

## Priority 1: PI Curve Raw Data (Temperature × Irradiance)
These papers contain PI curves measured at multiple temperatures.
Data must be digitized from published figures.

### P1-A: Torzillo & Vonshak (1994) ⭐ ✅ ACQUIRED (purchased from Elsevier CCC) ⭐ MOST CRITICAL
- **Title**: Effect of light and temperature on the photosynthetic activity
  of the cyanobacterium Spirulina platensis
- **Journal**: Biomass and Bioenergy, 6(6), 457-462
- **DOI**: 10.1016/0961-9534(94)00076-6
- **Data**: PI curves at multiple temperatures (20-40°C range)
- **Method**: O₂ electrode
- **Species**: Spirulina platensis
- **Status**: Closed access — requires figure digitization
- **Parameters extractable**: α(T), Pmax(T), convexity(T)

### P1-B: Vonshak & Guy (1992)
- **Title**: Photoadaptation, photoinhibition and productivity in the
  blue-green alga, Spirulina platensis grown outdoors
- **Journal**: Plant, Cell & Environment, 15(5), 613-616
- **DOI**: 10.1111/j.1365-3040.1992.tb01496.x
- **Data**: Outdoor PI curves with temperature variation
- **Cited by**: 117
- **Status**: Closed access

### P1-C: Torzillo et al. (1998) ✅ ACQUIRED
- **Title**: On-line monitoring of chlorophyll fluorescence to assess the
  extent of photoinhibition induced by high O₂ and low temperature
- **Journal**: Journal of Phycology, 34(5), 504-510
- **DOI**: 10.1046/j.1529-8817.1998.340504.x
- **Data**: PAM fluorescence-based PI parameters under temperature stress
- **Cited by**: 135
- **Status**: Closed access

## Priority 2: PBR Biomass & Night Loss Data
### P2-A: Torzillo et al. (1991a) ⭐ ✅ ACQUIRED
- **Title**: Effect of temperature on yield and night biomass loss in
  Spirulina platensis grown outdoors in tubular photobioreactors
- **Journal**: Journal of Applied Phycology, 3, 103-108
- **DOI**: 10.1007/bf00003691
- **Data**: Night biomass loss at 25°C vs 35°C, carbohydrate dynamics
- **Key values** (from literature):
  - Optimal temp: 35°C
  - Night loss at 35°C: ~5%, at 25°C: ~7.6%
  - Carbohydrate at 35°C: 22.40%, at 25°C: 28.83%
  - Net productivity: 35°C is 23% higher than 25°C
- **Cited by**: 137

### P2-B: Torzillo et al. (1991b)
- **Title**: Temperature as an important factor affecting productivity and
  night biomass loss in Spirulina platensis
- **Journal**: Bioresource Technology, 38(2-3), 95-100
- **DOI**: 10.1016/0960-8524(91)90137-9
- **Cited by**: 91

## Priority 3: D1 Repair Kinetics
### P3-A: Synechocystis FtsH protease studies
- **Key finding**: kd has low Ea (photophysical), kr has high Ea (enzymatic)
- **Temperature range**: 20°C, 30°C, 40°C at 600 µmol m⁻²s⁻¹
- **Parameters**: kd(T), kr(T), Fv/Fm recovery curves

### P3-B: Arthrospira UV-B studies
- **Species**: A. platensis strains 439 and D-0083
- **Key finding**: 56% loss in O₂ evolution and D1 after 180 min UV-B
- **Lincomycin effect**: Accelerated loss when repair blocked

## Priority 4: Structural Data
### P4-A: PBS-PSII Supercomplex (PDB: 8WQL)
- **Species**: Arthrospira sp. FACHB-439
- **Resolution**: 3.5 Å Cryo-EM
- **Structure**: 445-mer PBS-PSII supercomplex
- **Relevance**: σ_PSII estimation, antenna decoupling energetics

## Priority 5: CELSS Gas Exchange
### P5-A: Oguchi et al. (1987)
- **System**: 6L PBR with hollow fiber O₂ extraction
- **Data**: O₂ purity >46%, flow 100-150 ml/min continuous

## Data Processing Pipeline
1. Digitize PI curves from P1-A, P1-B, P1-C figures → CSV
2. Fit Ph10 model using eos_sensor.py framework
3. Extract α(T), Pmax(T), β(T) at each temperature
4. Build β(T) Arrhenius plot
5. Cross-validate with P2 biomass/carbohydrate data

## BONUS: Acquired papers not in original list

### B1: Tanaka et al. (2020) ✅ ACQUIRED (OPEN ACCESS CC-BY)
- **Title**: Low temperatures in dark period affect biomass productivity
  of a cyanobacterium Arthrospira platensis
- **Journal**: Algal Research, 52, 102132
- **DOI**: 10.1016/j.algal.2020.102132
- **Species**: A. platensis NIES-39 (Lake Chad isolate)
- **Data**:
  - Dark-period temp 10-35°C (5 levels) × productivity/night loss
  - Respiration rate regression: y = 0.168x - 0.721 (R²=0.968)
  - C:N ratio regression: y = -0.0722x + 6.8079 (R²=0.836)
  - Cross-species night loss comparison (8 species)
  - Net productivity comparison (3 species)
- **Key reference**: Jensen & Knutsen (1993) — photosynthetic recovery
  104% at 35°C but only 21.3% at 20°C after photoinhibition!
  (= direct evidence of kd/kr Arrhenius asymmetry)
