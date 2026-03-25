#!/usr/bin/env python3
"""
Midori-no-Tamaki: EOS-Thermal Extension Figure Generator
=========================================================
Generates all figures from Session 1 (2025-03-21/22).

Figures:
  1. Fig_arrhenius_kd_kr.png           — Arrhenius fitting results
  2. Fig_SAI_crossover_prediction.png  — SAI(T) fold-point & regime map
  3. Fig_cross_species_validation.png  — Cross-species EOS validation
  4. Fig_pH_CO2_interaction_map.png    — pH-CO2-Temperature interactions
  5. Fig_PBR_dynamics_simulation.png   — PBR feedback dynamics

Usage:
  python generate_thermal_figures.py [output_dir]

Dependencies: numpy, matplotlib
Authors: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys, os

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(OUTPUT_DIR, exist_ok=True)

R_gas = 8.314e-3  # kJ/mol/K


# =============================================================================
# FIGURE 3: Cross-Species Validation
# =============================================================================
def fig_cross_species_validation():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Rehder 2023: EOS-Thermal Cross-Species Validation\n'
                 '(Phaeodactylum tricornutum, Diatom — 6°C vs 15°C)',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel A: alpha temperature independence
    ax = axes[0, 0]
    bars = ax.bar([0, 1, 2.5, 3.5],
                  [1.2119, 1.2221, 1.2119, 1.2119*0.94],
                  yerr=[0.3216, 0.1613, 0.3216, 0.3],
                  color=['#2196F3', '#FF5722', '#2196F3', '#FF5722'],
                  alpha=0.7, width=0.8, edgecolor='black', linewidth=1.2)
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(['6C\n(accl)', '15C\n(accl)', '6C\n(at 6C)',
                        '6C cells\n(at 15C)'], fontsize=10)
    ax.set_ylabel('alpha (light use efficiency)', fontsize=12)
    ax.set_title('A - alpha is TEMPERATURE INDEPENDENT',
                 fontsize=12, fontweight='bold', color='green')
    ax.axhline(y=1.22, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.5, 0.95, 'Ratio: 1.01 (accl)\nRatio: 0.94 (acute)',
            transform=ax.transAxes, fontsize=11, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel B: Pmax acute vs acclimated
    ax = axes[0, 1]
    Pmax_6_at6 = [73.8, 78.0, 71.9, 72.1]
    Pmax_6_at15 = [135.6, 135.0, 135.9, 154.3]
    Pmax_15_at15 = [97.2, 77.1, 96.0]
    Pmax_15_at6 = [42.1, 36.8, 37.1]

    for p6, p15 in zip(Pmax_6_at6, Pmax_6_at15):
        ax.plot([6, 15], [p6, p15], 'o-', color='#2196F3', alpha=0.6, markersize=8)
    for p15, p6 in zip(Pmax_15_at15, Pmax_15_at6):
        ax.plot([15, 6], [p15, p6], 's-', color='#FF5722', alpha=0.6, markersize=8)

    ax.scatter([6], [np.mean(Pmax_6_at6)], marker='D', s=200, color='blue',
              zorder=5, edgecolors='black', linewidths=2, label='6C accl mean')
    ax.scatter([15], [np.mean(Pmax_15_at15)], marker='D', s=200, color='red',
              zorder=5, edgecolors='black', linewidths=2, label='15C accl mean')
    ax.set_xlabel('Measurement Temperature (C)', fontsize=12)
    ax.set_ylabel('Pmax (umol O2 / mg Chl / h)', fontsize=12)
    ax.set_title('B - Pmax: Acute vs Acclimated Response',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xlim(3, 18)
    ax.annotate('Acute: x1.90\n(Ea=47.5 kJ/mol)', xy=(10.5, 135), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                ha='center')
    ax.annotate('Acclimated:\nx1.22 only', xy=(10.5, 78), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                ha='center', style='italic')

    # Panel C: Ea comparison
    ax = axes[1, 0]
    species = ['Spirulina\n(our 2-pt fit)', 'Synechocystis\n(D1 synthesis)',
               'Synechocystis\n(repair rate)', 'Phaeodactylum\n(acute Pmax)',
               'Literature\ndefault (0.65eV)']
    Ea_vals = [240.1, 87.1, 74.2, 47.5, 63.0]
    colors = ['red', '#4CAF50', '#4CAF50', '#2196F3', '#FF9800']
    bars = ax.barh(range(len(species)), Ea_vals, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.2, height=0.6)
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species, fontsize=9)
    ax.set_xlabel('Ea (kJ/mol)', fontsize=12)
    ax.set_title('C - Activation Energy Across Species',
                 fontsize=12, fontweight='bold')
    ax.axvline(x=63, color='orange', linewidth=2.5, linestyle='--',
              label='0.65 eV default', alpha=0.8)
    ax.legend(fontsize=10, loc='lower right')
    for i, v in enumerate(Ea_vals):
        ax.text(v + 3, i, f'{v:.0f}', va='center', fontsize=11, fontweight='bold')
    ax.annotate('Overestimate\n(2-point fit)', xy=(240, 0), xytext=(180, 1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')

    # Panel D: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = """EOS-THERMAL: CROSS-SPECIES VALIDATION
========================================

[OK] alpha is temperature-INDEPENDENT
   Spirulina: confirmed (Torzillo)
   Synechocystis: confirmed (Allakhverdiev)
   Phaeodactylum: ratio = 1.01  <-- NEW

[OK] kd is temperature-INDEPENDENT
   Synechocystis: CV = 0.9% (10-34C)

[OK] kr (via Pmax) is Arrhenius-type
   Synechocystis: Ea ~ 74-87 kJ/mol
   Phaeodactylum: Ea ~ 48 kJ/mol (acute)
   Literature default: 63 kJ/mol (0.65 eV)

[OK] Acclimation != Acute response
   Acute Pmax ratio:      x1.90
   Acclimated Pmax ratio: x1.22
   --> SAI captures the DIFFERENCE

[OK] Denaturation threshold
   Synechocystis: Tm ~ 42-44C (Ueno 2016)

CONCLUSION:
  b_env = min(b_thermal, b_carbon)
  --> UNIVERSAL across prokaryotes & eukaryotes"""
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9.8,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig_cross_species_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 4: pH-CO2 Interaction Map
# =============================================================================
def fig_pH_CO2_interaction():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Spirulina pH-CO2-Temperature Interaction Map\n'
                 'for EOS-thermal b_carbon Channel',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel A: Carbonate equilibrium vs pH
    ax = axes[0, 0]
    pH_range = np.linspace(6, 13, 500)
    H = 10**(-pH_range)
    K1 = 4.3e-7; K2 = 4.7e-11
    alpha0 = 1 / (1 + K1/H + K1*K2/H**2)
    alpha1 = 1 / (H/K1 + 1 + K2/H)
    alpha2 = 1 / (H**2/(K1*K2) + H/K2 + 1)

    ax.plot(pH_range, alpha0*100, 'r-', linewidth=2.5, label='CO$_2$(aq)')
    ax.plot(pH_range, alpha1*100, 'g-', linewidth=2.5, label='HCO$_3^-$')
    ax.plot(pH_range, alpha2*100, 'b-', linewidth=2.5, label='CO$_3^{2-}$')
    ax.axvspan(8.5, 10.5, alpha=0.15, color='green', label='Spirulina optimal')
    ax.axvspan(11.5, 13, alpha=0.15, color='red', label='pH>12: death')
    ax.axvline(x=8.95, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=12.0, color='red', linestyle='--', alpha=0.7)
    ax.annotate('Start pH\n(Kobayashi)', xy=(8.95, 85), fontsize=8, ha='center')
    ax.annotate('Growth\nstop!', xy=(12.0, 70), fontsize=8, ha='center', color='red')
    ax.set_xlabel('pH', fontsize=12); ax.set_ylabel('Species fraction (%)', fontsize=12)
    ax.set_title('A - Carbonate Equilibrium (25C)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='center right'); ax.set_xlim(6, 13); ax.set_ylim(0, 100)

    # Panel B: Available carbon vs pH
    ax = axes[0, 1]
    DIC_mM = 200
    HCO3_mM = DIC_mM * alpha1
    CO2_mM = DIC_mM * alpha0
    CO3_mM = DIC_mM * alpha2

    ax.semilogy(pH_range, HCO3_mM, 'g-', linewidth=2.5, label='[HCO$_3^-$]')
    ax.semilogy(pH_range, CO2_mM, 'r-', linewidth=2.5, label='[CO$_2$(aq)]')
    ax.semilogy(pH_range, CO3_mM, 'b-', linewidth=2.5, label='[CO$_3^{2-}$]')
    ax.axhline(y=0.03, color='orange', linestyle='--', linewidth=2,
              label='Km(RuBisCO) ~30 uM')
    ax.axvspan(8.5, 10.5, alpha=0.15, color='green')
    ax.set_xlabel('pH', fontsize=12); ax.set_ylabel('Concentration (mM)', fontsize=12)
    ax.set_title('B - Available Carbon in Zarrouk Medium',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xlim(6, 13); ax.set_ylim(1e-4, 300)

    # Panel C: Temperature effects
    ax = axes[1, 0]
    T_range = np.linspace(10, 45, 100)
    T_K = T_range + 273.15
    KH_25 = 3.4e-2; delta_H = -2400; R = 8.314
    KH_T = KH_25 * np.exp(-delta_H/R * (1/T_K - 1/298.15))
    pK1 = 6.352 - 0.0152 * T_range; K1_T = 10**(-pK1)
    KH_norm = KH_T / KH_25
    K1_norm = K1_T / (10**(-6.352 + 0.0152*25))
    Ea_r = 63.0
    kr_norm = np.exp((Ea_r/8.314e-3) * (1/298.15 - 1/T_K))
    kr_norm = kr_norm / kr_norm[np.argmin(np.abs(T_range-25))]

    ax.plot(T_range, KH_norm, 'r-', linewidth=2.5, label='CO$_2$ solubility (KH)')
    ax.plot(T_range, K1_norm, 'g-', linewidth=2.5, label='K1 (HCO$_3^-$ formation)')
    ax.plot(T_range, kr_norm, 'purple', linewidth=2.5, linestyle='--',
           label='kr(T) (D1 repair)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=35, color='orange', linestyle='--', alpha=0.5)
    ax.annotate('T_opt\nSpirulina', xy=(35, 0.5), fontsize=9, color='orange')
    ax.set_xlabel('Temperature (C)', fontsize=12)
    ax.set_ylabel('Relative to 25C', fontsize=12)
    ax.set_title('C - Temperature Effects: Competing Processes',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 5)

    # Panel D: Feedback diagram
    ax = axes[1, 1]
    ax.axis('off')
    diagram = """  CONTROL INPUTS          PHYSICAL STATES          OUTPUTS
  ==============          ===============          =======

  Light (I) --------+
                    v
             +-----------+
             | Pgross    |--- CO2 consumed --+
             | (I,T,Ci)  |                   |
             +-----------+                   v
                    ^                   +----------+
  Temp (T) --------+                   | pH shift |
    |              |                   +----+-----+
    |       +------+------+                 |
    |       | R(T)        |                 v
    |       | respiration |-- CO2 ----> Ci(pH,DIC)
    |       +-------------+  produced       |
    |                                       |
    +--- CO2 solubility(T) --------------->|
                                            |
  CO2 gas ---- kLa -- CO2 supply --------->|
  (control)                                 |
                                            v
                                     +-----------+
  Nitrogen --------------------------| b_env  |
  (control)                          | = min(    |
                                     |  b_thermal,|
                                     |  b_therm, |
                                     |  b_carbon)|
                                     +-----+-----+
                                           |
                                           v
                                      NET GROWTH
                                     = Pgross*eta
                                       - R(T)

  OPERATOR KNOBS:  I, T, CO2_gas, N, pH
  SENSORS:         PAM(a,Pmax), T, pH
  DIAGNOSIS:       SAI = SAI_T + SAI_C + SAI_N"""
    ax.text(0.0, 0.98, diagram, transform=ax.transAxes, fontsize=7.5,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('D - System Feedback Architecture',
                 fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'Fig_pH_CO2_interaction_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Generating EOS-Thermal Extension figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print("[3/5] Cross-species validation...")
    fig_cross_species_validation()

    print("[4/5] pH-CO2 interaction map...")
    fig_pH_CO2_interaction()

    print()
    print("NOTE: Fig 1-2 (Arrhenius) are in soba/fit_arrhenius_kd_kr.py")
    print("NOTE: Fig 5 (PBR dynamics) is in soba/pbr_dynamics.py")
    print()
    print("All done!")
