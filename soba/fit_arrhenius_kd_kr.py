#!/usr/bin/env python3
"""
fit_arrhenius_kd_kr.py - Fit D1 damage/repair Arrhenius parameters
=================================================================

Physics model:
    dA/dt = -kd*I*A + kr*(1-A)
    
    A(t) = fraction of active PSII remaining
    kd   = D1 damage rate constant (per photon per unit time)
    kr   = D1 repair rate constant (per unit time)
    I    = irradiance (umol m-2 s-1)
    
    Analytical solution:
    A(t) = A_inf + (1 - A_inf) * exp(-lambda * t)
    where:
        lambda = kd*I + kr          (total decay rate)
        A_inf  = kr / (kd*I + kr)   (steady-state fraction)

Arrhenius model:
    kd(T) = kd0 * exp(-Ea_d / RT)   (damage: LOW Ea, weak T-dependence)
    kr(T) = kr0 * exp(-Ea_r / RT)   (repair: HIGH Ea, strong T-dependence)
    
    K(T) = kd/kr = (kd0/kr0) * exp((Ea_r - Ea_d) / RT)

Data sources:
    Torzillo & Vonshak (1994) Fig.5: photoinhibition kinetics at 25, 35, 40 C
    Jensen & Knutsen (1993): recovery at 20 C (21.3%) vs 35 C (104%)

Author: M. Iizumi & T. Iizumi (Miosync, Inc.)
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
R_gas = 8.314e-3  # kJ mol-1 K-1
I_HPFD = 2500.0   # umol m-2 s-1 (photoinhibitory irradiance in Torzillo 1994)

# ============================================================
# DATA: Torzillo & Vonshak (1994) Fig.5
# Photoinhibition kinetics at 2500 umol m-2 s-1
# Values: fraction of control activity remaining
# ============================================================
time_min = np.array([0, 10, 20, 30, 40, 50, 60], dtype=float)

data = {
    25: np.array([1.00, 0.80, 0.65, 0.50, 0.38, 0.28, 0.22]),
    35: np.array([1.00, 0.88, 0.75, 0.65, 0.55, 0.50, 0.456]),
    40: np.array([1.00, 0.85, 0.72, 0.60, 0.50, 0.43, 0.39]),
}

# ============================================================
# MODEL FUNCTION
# ============================================================
def A_model(t, A_inf, lam):
    """D1 damage-repair analytical solution.
    
    A(t) = A_inf + (1 - A_inf) * exp(-lam * t)
    
    Parameters:
        A_inf : steady-state active fraction = kr / (kd*I + kr)
        lam   : total rate = kd*I + kr [min-1]
    """
    return A_inf + (1.0 - A_inf) * np.exp(-lam * t)


# ============================================================
# FIT EACH TEMPERATURE
# ============================================================
print("=" * 70)
print("D1 DAMAGE-REPAIR KINETICS FITTING")
print("Model: dA/dt = -kd*I*A + kr*(1-A)")
print(f"Irradiance: {I_HPFD} umol m-2 s-1")
print("=" * 70)

results = {}

for T_C, A_data in sorted(data.items()):
    T_K = T_C + 273.15
    
    # Fit A_inf and lambda
    popt, pcov = curve_fit(
        A_model, time_min, A_data,
        p0=[0.2, 0.05],       # initial guess
        bounds=([0, 0], [1, 1]),  # physical bounds
        maxfev=10000
    )
    
    A_inf_fit, lam_fit = popt
    perr = np.sqrt(np.diag(pcov))
    
    # Extract kd and kr from fitted parameters
    # A_inf = kr / (kd*I + kr)  =>  kr = A_inf * (kd*I + kr) = A_inf * lam
    # lam = kd*I + kr
    kr = A_inf_fit * lam_fit              # min-1
    kd_I = lam_fit - kr                    # min-1 (= kd * I)
    kd = kd_I / I_HPFD                    # min-1 per (umol m-2 s-1)
    
    # Residuals
    A_pred = A_model(time_min, A_inf_fit, lam_fit)
    residuals = A_data - A_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((A_data - np.mean(A_data))**2)
    r2 = 1 - ss_res / ss_tot
    
    results[T_C] = {
        'T_K': T_K,
        'A_inf': A_inf_fit,
        'lambda': lam_fit,
        'kr': kr,
        'kd_I': kd_I,
        'kd': kd,
        'K': kd_I / kr if kr > 0 else float('inf'),  # K = kd*I/kr
        'r2': r2,
    }
    
    print(f"\n--- T = {T_C} C ({T_K:.1f} K) ---")
    print(f"  A_inf  = {A_inf_fit:.4f} +/- {perr[0]:.4f}")
    print(f"  lambda = {lam_fit:.5f} +/- {perr[1]:.5f} min-1")
    print(f"  kr     = {kr:.6f} min-1")
    print(f"  kd*I   = {kd_I:.6f} min-1")
    print(f"  kd     = {kd:.2e} min-1 / (umol m-2 s-1)")
    print(f"  K(T)   = kd*I/kr = {results[T_C]['K']:.3f}")
    print(f"  R2     = {r2:.6f}")


# ============================================================
# ARRHENIUS FIT: ln(k) vs 1/T
# ============================================================
print("\n" + "=" * 70)
print("ARRHENIUS ANALYSIS")
print("=" * 70)

temps_C = sorted(results.keys())
temps_K = np.array([results[T]['T_K'] for T in temps_C])
inv_T = 1.0 / temps_K

kr_vals = np.array([results[T]['kr'] for T in temps_C])
kd_vals = np.array([results[T]['kd'] for T in temps_C])
K_vals = np.array([results[T]['K'] for T in temps_C])

ln_kr = np.log(kr_vals)
ln_kd = np.log(kd_vals)
ln_K = np.log(K_vals)

# Linear fit: ln(k) = ln(k0) - Ea/(R*T)  =>  ln(k) = a + b*(1/T)
# slope b = -Ea/R  =>  Ea = -b * R

# Fit kr (repair)
coeff_kr = np.polyfit(inv_T, ln_kr, 1)
Ea_r = -coeff_kr[0] * R_gas  # kJ/mol
kr0 = np.exp(coeff_kr[1])

print(f"\nRepair rate kr(T):")
print(f"  Ea_repair = {Ea_r:.1f} kJ/mol")
print(f"  kr0       = {kr0:.4e} min-1")
print(f"  kr(T) = {kr0:.4e} * exp(-{Ea_r:.1f} / (R*T))")

# Fit kd (damage)
coeff_kd = np.polyfit(inv_T, ln_kd, 1)
Ea_d = -coeff_kd[0] * R_gas
kd0 = np.exp(coeff_kd[1])

print(f"\nDamage rate kd(T):")
print(f"  Ea_damage = {Ea_d:.1f} kJ/mol")
print(f"  kd0       = {kd0:.4e} min-1 / (umol m-2 s-1)")
print(f"  kd(T) = {kd0:.4e} * exp(-{Ea_d:.1f} / (R*T))")

# Delta Ea
delta_Ea = Ea_r - Ea_d
print(f"\nDelta Ea (Ea_repair - Ea_damage) = {delta_Ea:.1f} kJ/mol")
print(f"  -> {'POSITIVE (repair more T-sensitive, as expected!)' if delta_Ea > 0 else 'NEGATIVE (unexpected)'}")

# Fit K(T) directly
coeff_K = np.polyfit(inv_T, ln_K, 1)
Ea_K = -coeff_K[0] * R_gas  # should be -(Ea_r - Ea_d) = -delta_Ea
K0 = np.exp(coeff_K[1])

print(f"\nK(T) = kd*I/kr (damage-to-repair ratio at I={I_HPFD}):")
print(f"  Ea_K      = {-Ea_K:.1f} kJ/mol (= delta_Ea)")
print(f"  K0        = {K0:.4e}")

# ============================================================
# PREDICT b_thermal(T)
# ============================================================
print("\n" + "=" * 70)
print("BETA_THERMAL(T) PREDICTION")
print("=" * 70)

# From original EOS: beta_pred(alpha) via scaling law
# b_thermal(T) is the temperature-modified effective beta
# Using: A_inf(T) = 1 / (1 + K(T)*I)
# At reference conditions (T_opt, moderate I):
# b_thermal(T) / b_thermal(T_opt) = A_inf(T) / A_inf(T_opt)

T_range = np.linspace(5, 50, 100)
T_range_K = T_range + 273.15

# Predict K(T) over full range
K_pred = K0 * np.exp(-(-Ea_K) / (R_gas * T_range_K))

# At a reference irradiance (e.g. 500 umol for operational conditions)
I_ref = 500.0
K_at_Iref = K_pred * (I_ref / I_HPFD)  # scale K from 2500 to 500

# Effective photosynthetic capacity relative to optimal
# A_inf(T) = 1 / (1 + K(T)*I)
A_inf_pred = 1.0 / (1.0 + K_at_Iref)

# Find T_opt index
idx_opt = np.argmax(A_inf_pred)
T_opt_pred = T_range[idx_opt]

print(f"At I_ref = {I_ref} umol m-2 s-1:")
for T_check in [10, 15, 20, 25, 30, 35, 40, 45]:
    idx = np.argmin(np.abs(T_range - T_check))
    print(f"  T={T_check:2d}C: K={K_at_Iref[idx]:.3f}, A_inf={A_inf_pred[idx]:.3f}, beta_ratio={A_inf_pred[idx]/A_inf_pred[np.argmin(np.abs(T_range-35))]:.3f}")


# ============================================================
# PLOT
# ============================================================
out_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(os.path.dirname(out_dir), 'figures')
os.makedirs(fig_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('D1 Damage-Repair Arrhenius Analysis\nTorzillo & Vonshak (1994)', fontsize=14, fontweight='bold')

# Panel A: Kinetics fit
ax = axes[0, 0]
colors = {25: '#378ADD', 35: '#1D9E75', 40: '#D85A30'}
t_fine = np.linspace(0, 65, 200)
for T_C in sorted(data.keys()):
    ax.scatter(time_min, data[T_C] * 100, color=colors[T_C], s=50, zorder=5, label=f'{T_C} C data')
    A_fit = A_model(t_fine, results[T_C]['A_inf'], results[T_C]['lambda'])
    ax.plot(t_fine, A_fit * 100, color=colors[T_C], linewidth=2, 
            label=f'{T_C} C fit (R2={results[T_C]["r2"]:.4f})')
ax.set_xlabel('Time in HPFD (min)')
ax.set_ylabel('Activity remaining (%)')
ax.set_title('A: Photoinhibition kinetics at 2500 umol')
ax.legend(fontsize=8)
ax.set_ylim(15, 105)
ax.grid(alpha=0.3)

# Panel B: Arrhenius plot
ax = axes[0, 1]
ax.scatter(inv_T * 1e3, ln_kr, color='#1D9E75', s=80, zorder=5, label='kr (repair)')
ax.scatter(inv_T * 1e3, ln_kd, color='#D85A30', s=80, zorder=5, label='kd (damage)')

inv_T_fine = np.linspace(min(inv_T) - 0.05e-3, max(inv_T) + 0.05e-3, 100)
ax.plot(inv_T_fine * 1e3, coeff_kr[0] * inv_T_fine + coeff_kr[1], '--', color='#1D9E75', linewidth=2,
        label=f'kr: Ea={Ea_r:.1f} kJ/mol')
ax.plot(inv_T_fine * 1e3, coeff_kd[0] * inv_T_fine + coeff_kd[1], '--', color='#D85A30', linewidth=2,
        label=f'kd: Ea={Ea_d:.1f} kJ/mol')

ax.set_xlabel('1/T (10^-3 K^-1)')
ax.set_ylabel('ln(k)')
ax.set_title(f'B: Arrhenius plot  (dEa = {delta_Ea:.1f} kJ/mol)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Add second x-axis for temperature
ax2 = ax.twiny()
temp_ticks_C = [25, 30, 35, 40]
temp_ticks_inv = [1.0 / (T + 273.15) * 1e3 for T in temp_ticks_C]
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(temp_ticks_inv)
ax2.set_xticklabels([f'{T}C' for T in temp_ticks_C])

# Panel C: K(T) over full temperature range
ax = axes[1, 0]
ax.plot(T_range, K_at_Iref, color='#534AB7', linewidth=2.5)
ax.scatter([25, 35, 40], [results[T]['K'] * (I_ref / I_HPFD) for T in [25, 35, 40]], 
           color='#534AB7', s=80, zorder=5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='K=1 (balanced)')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel(f'K(T) at I={I_ref} umol')
ax.set_title('C: Damage/repair ratio K(T)')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel D: b_thermal relative to optimal
ax = axes[1, 1]
ax.plot(T_range, A_inf_pred / A_inf_pred[np.argmin(np.abs(T_range - 35))], 
        color='#993C1D', linewidth=2.5, label=f'beta_ratio at I={I_ref}')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=35, color='#1D9E75', linestyle=':', alpha=0.5, label='T_opt = 35C')
ax.fill_between(T_range, 0, A_inf_pred / A_inf_pred[np.argmin(np.abs(T_range - 35))],
                where=(T_range < 20), alpha=0.15, color='#378ADD', label='Cold stress regime')
ax.fill_between(T_range, 0, A_inf_pred / A_inf_pred[np.argmin(np.abs(T_range - 35))],
                where=(T_range > 42), alpha=0.15, color='#D85A30', label='Hot stress regime')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('b_thermal(T) / b_thermal(35C)')
ax.set_title('D: b_thermal(T) closed form')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(fig_dir, 'Fig_arrhenius_kd_kr.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")

# Also save parameters to CSV
csv_path = os.path.join(os.path.dirname(out_dir), 'data', 'arrhenius_fitted_params.csv')
with open(csv_path, 'w') as f:
    f.write('# Arrhenius parameters from D1 damage-repair fitting\n')
    f.write('# Source: Torzillo & Vonshak (1994) Fig.5, 2500 umol m-2 s-1\n')
    f.write('# Model: dA/dt = -kd*I*A + kr*(1-A)\n')
    f.write('#\n')
    f.write('parameter,value,unit,notes\n')
    f.write(f'Ea_repair,{Ea_r:.2f},kJ_mol-1,HIGH - repair is T-sensitive\n')
    f.write(f'Ea_damage,{Ea_d:.2f},kJ_mol-1,LOW - damage is photophysical\n')
    f.write(f'delta_Ea,{delta_Ea:.2f},kJ_mol-1,Ea_r - Ea_d positive as expected\n')
    f.write(f'kr0,{kr0:.6e},min-1,pre-exponential repair\n')
    f.write(f'kd0,{kd0:.6e},min-1_per_umol_m-2_s-1,pre-exponential damage\n')
    f.write(f'K0,{K0:.6e},dimensionless,pre-exponential K at I=2500\n')
    for T_C in sorted(results.keys()):
        r = results[T_C]
        f.write(f'kr_{T_C}C,{r["kr"]:.6f},min-1,fitted repair rate at {T_C}C\n')
        f.write(f'kd_{T_C}C,{r["kd"]:.2e},min-1_per_umol,fitted damage rate at {T_C}C\n')
        f.write(f'K_{T_C}C,{r["K"]:.4f},dimensionless,K at {T_C}C and 2500 umol\n')
        f.write(f'A_inf_{T_C}C,{r["A_inf"]:.4f},fraction,steady-state active PSII at {T_C}C\n')
        f.write(f'R2_{T_C}C,{r["r2"]:.6f},dimensionless,goodness of fit\n')
print(f'Parameters saved: {csv_path}')

plt.show()
