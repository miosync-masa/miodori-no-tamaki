#!/usr/bin/env python3
"""
pbr_dynamics.py
===============
Fig. 8 of the manuscript: Spirulina PBR dynamic simulation illustrating
the feedback loop (photosynthetic CO2 uptake -> DIC decline -> pH rise -> carbon
capacity), with the state-estimation layer evaluated along the trajectory.

State variables: DIC(t), Biomass(t); alkalinity is conserved.
Derived:         pH(t) (charge balance), b_thermal, b_carbon, b_env, beta_eff, mu(t)

Photosynthesis uses the main-text formulation (Eqs. 6-7, 9, 12) with the Part I
double-tanh PI curve:
    b_env    = min(1, b_thermal(T), b_carbon(pH, DIC, T))
    beta_eff = beta_ref / b_env
    P_gross  = Pmax tanh(alpha I / Pmax) tanh((Pmax / (beta_eff I))**gamma0) * biomass
i.e. the environment acts on photoinhibition susceptibility (beta), not as a
multiplier on P_gross. beta_ref = alpha / S_REF with S_REF the median S = alpha/beta
of the Part I Ph10 dataset (dimensionless, so independent of the P units).

CO2 control is a pH-stat: the sparging valve opens proportionally between
pH 9.0 and 10.0 (inside the 8.5-10 range held by Kobayashi & Fujita 1996), with
CO2 saturation from Henry's law at 5% CO2 and kLa_max = 50 h-1 at full opening.

Integration: scipy solve_ivp (LSODA, rtol 1e-8, max_step 0.05 h); pH from the
charge balance by bisection (brentq). The explicit-Euler scheme with dt = 0.1 h
used previously produced spurious pH/b_carbon oscillations in the CO2-controlled
35 C scenario.

Usage (from the repository root):
  python soba2/pbr_dynamics.py

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eos_thermal_model import (R_GAS, T0_K, THERMAL_DEFAULT, KM_EFF_MM, K_H,  # noqa: E402
                               b_thermal, b_carbon, b_env, beta_eff, carbonate_fractions,
                               ph10_gross, pH_from_DIC_alk)

OUT = os.path.join(HERE, 'output')
S_REF = 12.48     # median alpha/beta of the 1,808 Part I Ph10 curves
P_CO2_ATM = 0.05  # 5% CO2 in the sparging gas (Kobayashi & Fujita 1996)


# === Process rates ===
def light_avg(I, biomass_gL, params):
    """Depth-averaged irradiance (Beer-Lambert)."""
    tau = params['k_ext'] * biomass_gL * params['path_length']
    return I * (1 - np.exp(-tau)) / (tau + 1e-10)


def photosynthesis_rate(I, T_C, pH, DIC_mM, biomass_gL, params):
    """Gross photosynthesis [mM C / h] and the diagnostic quantities."""
    I_avg = light_avg(I, biomass_gL, params)
    Pmax = params['Pmax_opt']                         # mM C / g biomass / h
    alpha = Pmax / params['Ik']
    beta_ref = alpha / params['S_ref']
    bt = b_thermal(T_C, **params['thermal'])
    bc = b_carbon(pH, DIC_mM, T_C, Km_mM=params['Km_mM'])
    be = b_env(bt, bc)
    b_eff = beta_eff(beta_ref, be)
    Pgross = ph10_gross(I_avg, alpha, b_eff, Pmax) * biomass_gL
    return Pgross, bt, bc, be, b_eff / beta_ref, I_avg


def respiration_rate(T_C, biomass_gL, params):
    """Dark respiration [mM C / h]."""
    T_K, T_ref_K = T_C + T0_K, params['T_ref'] + T0_K
    return params['R_ref'] * np.exp((params['Ea_resp'] / R_GAS) * (1 / T_ref_K - 1 / T_K)) * biomass_gL


def pH_stat_opening(pH, params):
    """pH-stat valve opening u in [0, 1]: closed below pH_lo, fully open above pH_hi (proportional band)."""
    lo, hi = params['pH_stat']
    return float(np.clip((pH - lo) / (hi - lo), 0.0, 1.0))


def CO2_supply(pH, DIC_mM, T_C, params):
    """CO2 gas transfer [mM C / h] under pH-stat control; saturation from Henry's law at the sparging pCO2."""
    if not params['CO2_on']:
        return 0.0
    CO2_sat = 1e3 * K_H(T_C) * params['pCO2_atm']    # mM
    a0, _, _ = carbonate_fractions(pH, T_C)
    return pH_stat_opening(pH, params) * params['kLa_max'] * (CO2_sat - DIC_mM * a0)


def growth_rate(Pgross, R, biomass_gL, params):
    """Net specific growth rate [h^-1]."""
    net_C = Pgross * params['eta_growth'] - R
    mu = net_C / (params['C_per_biomass'] * biomass_gL + 1e-10)
    return max(mu, -0.01)


# === Simulation ===
def simulate_PBR(params, T_hours=300, dt_out=0.1, rtol=1e-8, max_step=0.05):
    T_C, I0, alk = params['temperature'], params['irradiance'], params['alkalinity']

    def irradiance(t):
        if params.get('light_cycle', False):
            return I0 if (t % 24) < 16 else 0.0
        return I0

    def rhs(t, y):
        DIC, bio = max(y[0], 0.1), max(y[1], 0.01)
        pH = pH_from_DIC_alk(DIC, alk, T_C)
        Pg, *_ = photosynthesis_rate(irradiance(t), T_C, pH, DIC, bio, params)
        R = respiration_rate(T_C, bio, params)
        dDIC = -Pg + R + CO2_supply(pH, DIC, T_C, params)
        if y[0] <= 0.1 and dDIC < 0:
            dDIC = 0.0
        return [dDIC, growth_rate(Pg, R, bio, params) * bio]

    t_eval = np.arange(0, T_hours + dt_out / 2, dt_out)
    sol = solve_ivp(rhs, (0, T_hours), [params['DIC_init'], params['biomass_init']],
                    method='LSODA', t_eval=t_eval, rtol=rtol, atol=1e-10, max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    DIC, bio = np.maximum(sol.y[0], 0.1), np.maximum(sol.y[1], 0.01)
    out = {k: np.zeros_like(sol.t) for k in ['pH', 'Pgross', 'R', 'b_thermal', 'b_carbon', 'b_env',
                                             'beta_ratio', 'I_avg', 'mu']}
    for i, t in enumerate(sol.t):
        pH = pH_from_DIC_alk(DIC[i], alk, T_C)
        Pg, bt, bc, be, br, Ia = photosynthesis_rate(irradiance(t), T_C, pH, DIC[i], bio[i], params)
        R = respiration_rate(T_C, bio[i], params)
        for k, v in zip(['pH', 'Pgross', 'R', 'b_thermal', 'b_carbon', 'b_env', 'beta_ratio', 'I_avg', 'mu'],
                        [pH, Pg, R, bt, bc, be, br, Ia, growth_rate(Pg, R, bio[i], params)]):
            out[k][i] = v
    out.update(t=sol.t, DIC=DIC, biomass=bio)
    return out


# === Parameter set ===
base_params = {
    # Culture conditions
    'DIC_init': 200.0,       # mM (Zarrouk medium: 16.8 g/L NaHCO3)
    'alkalinity': 200.0,     # mM (from NaHCO3)
    'biomass_init': 0.1,     # g/L
    'temperature': 35.0,     # °C
    'irradiance': 250.0,     # µmol/m²/s

    # Photosynthesis parameters
    'Pmax_opt': 2.0,         # mM C / g biomass / h
    'Ik': 150.0,             # µmol/m²/s (light saturation, = Pmax / alpha)
    'S_ref': S_REF,          # alpha / beta_ref
    'k_ext': 0.5,            # L/g/cm (extinction coefficient; generic, not calibrated to measured Arthrospira optics)
    'path_length': 4.0,      # cm (tube diameter)
    'thermal': dict(THERMAL_DEFAULT),
    'Km_mM': KM_EFF_MM,

    # Respiration
    'R_ref': 0.3,            # mM C / g biomass / h at T_ref
    'T_ref': 35.0,           # °C
    'Ea_resp': 48.8,         # kJ/mol (Torzillo & Vonshak 1994)

    # CO2 supply
    'CO2_on': False,
    'kLa_max': 50.0,         # h^-1, gas transfer coefficient at full valve opening
    'pCO2_atm': P_CO2_ATM,
    'pH_stat': (9.0, 10.0),  # proportional band inside Kobayashi's controlled range 8.5-10

    # Growth
    'eta_growth': 0.6,       # allocation to growth
    'C_per_biomass': 40.0,   # mM C per g biomass (~480 mg C/g)

    'light_cycle': False,    # continuous light
}

SCENARIOS = [
    ('s1', '25°C, no CO₂ (Kobayashi replication)', {'CO2_on': False, 'temperature': 25.0}),
    ('s2', '25°C, CO₂ controlled', {'CO2_on': True, 'temperature': 25.0}),
    ('s3', '35°C, CO₂ controlled', {'CO2_on': True, 'temperature': 35.0}),
    ('s4', '25°C, no CO₂, high light (500 µE)', {'CO2_on': False, 'temperature': 25.0, 'irradiance': 500.0}),
]


def summarise(res):
    is_carbon = res['b_carbon'] < res['b_thermal']
    k = int(np.argmax(res['pH'] > 12.0)) if np.any(res['pH'] > 12.0) else None
    switch = res['t'][np.where(np.diff(is_carbon.astype(int)) != 0)[0] + 1].round(1).tolist()
    dpH = np.diff(res['pH'])
    return {
        'max_biomass_gL': float(res['biomass'].max()), 'final_biomass_gL': float(res['biomass'][-1]),
        'final_pH': float(res['pH'][-1]), 'max_pH': float(res['pH'].max()),
        't_pH_above_12_h': float(res['t'][k]) if k is not None else None,
        'final_DIC_mM': float(res['DIC'][-1]),
        'min_b_carbon': float(res['b_carbon'].min()), 'b_thermal': float(res['b_thermal'][0]),
        'min_b_env': float(res['b_env'].min()), 'max_beta_ratio': float(res['beta_ratio'].max()),
        'carbon_dominant_fraction': float(is_carbon.mean()), 'switch_times_h': switch,
        'final_mu_per_day': float(res['mu'][-1] * 24),
        'pH_direction_changes': int(np.sum(np.diff(np.sign(dpH[np.abs(dpH) > 1e-9])) != 0)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print('Running simulations...')
    results = {key: simulate_PBR({**base_params, **over}) for key, _, over in SCENARIOS}
    labels = {key: lab for key, lab, _ in SCENARIOS}

    # integration-step check: rerun with a looser tolerance and coarser max_step
    check = {key: simulate_PBR({**base_params, **over}, rtol=1e-6, max_step=0.5)
             for key, _, over in SCENARIOS}
    summary = {key: summarise(res) for key, res in results.items()}
    for key in results:
        summary[key]['step_check_max_abs_diff'] = {
            'biomass_gL': float(np.max(np.abs(results[key]['biomass'] - check[key]['biomass']))),
            'pH': float(np.max(np.abs(results[key]['pH'] - check[key]['pH'])))}
    with open(os.path.join(OUT, 'pbr_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    # === Plotting ===
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Spirulina PBR Dynamic Simulation: Feedback Loop Behavior\n'
                 'b_env = min(1, b_thermal, b_carbon); β_eff = β_ref/b_env',
                 fontsize=14, fontweight='bold', y=0.99)
    colors = {'s1': '#e74c3c', 's2': '#3498db', 's3': '#2ecc71', 's4': '#e67e22'}
    styles = {'s1': '-', 's2': '--', 's3': '-', 's4': '-.'}

    ax = axes[0, 0]
    for key, res in results.items():
        ax.plot(res['t'], res['biomass'], color=colors[key], ls=styles[key], lw=2, label=labels[key])
    ax.set_ylabel('Biomass (g/L)', fontsize=12)
    ax.set_title('A - Biomass Growth', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.axhline(y=1.2, color=colors['s1'], linestyle=':', alpha=0.5)
    ax.axhline(y=2.2, color=colors['s2'], linestyle=':', alpha=0.5)
    ax.text(250, 1.3, '1.2 g/L\n(Kobayashi)', fontsize=7, color=colors['s1'])

    ax = axes[0, 1]
    for key, res in results.items():
        ax.plot(res['t'], res['pH'], color=colors[key], ls=styles[key], lw=2)
    ax.axhspan(8.5, 10.5, alpha=0.1, color='green', label='Optimal pH range')
    ax.axhline(y=12.0, color='red', linestyle='--', alpha=0.7, label='pH 12: death zone')
    ax.set_ylabel('pH', fontsize=12)
    ax.set_title('B - pH Dynamics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(8, 13)

    ax = axes[1, 0]
    for key, res in results.items():
        ax.plot(res['t'], res['DIC'], color=colors[key], ls=styles[key], lw=2)
    ax.set_ylabel('DIC (mM)', fontsize=12)
    ax.set_title('C - Dissolved Inorganic Carbon', fontsize=12, fontweight='bold')

    ax = axes[1, 1]
    for key, res in results.items():
        ax.plot(res['t'], res['b_carbon'], color=colors[key], lw=2, linestyle='-')
        ax.plot(res['t'], np.minimum(res['b_thermal'], 1.0), color=colors[key], lw=1.5, linestyle='--', alpha=0.6)
    ax.set_ylabel('Capacity factor b', fontsize=12)
    ax.set_title('D - b_carbon (solid) vs b_thermal (dashed)', fontsize=12, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(5, 0.52, 'Bottleneck threshold', fontsize=8, color='gray')
    ax.set_ylim(0, 1.1)

    ax = axes[2, 0]
    res = results['s1']                      # shading: lowest capacity in the uncontrolled 25 C culture
    is_carbon = res['b_carbon'] < res['b_thermal']
    ax.fill_between(res['t'], 0, 1, where=is_carbon, alpha=0.2, color='blue', label='S1: b_carbon lowest')
    ax.fill_between(res['t'], 0, 1, where=~is_carbon, alpha=0.2, color='orange', label='S1: b_thermal lowest')
    for key in ['s1', 's3']:
        ax.plot(results[key]['t'], results[key]['b_env'], color=colors[key], lw=2, label=labels[key][:30])
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('b_env (bottleneck capacity)', fontsize=12)
    ax.set_title('E - Bottleneck Switching', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower left')
    ax.set_ylim(0, 1.1)

    ax = axes[2, 1]
    for key, res in results.items():
        ax.plot(res['t'], res['mu'] * 24, color=colors[key], ls=styles[key], lw=2)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Growth rate (day⁻¹)', fontsize=12)
    ax.set_title('F - Specific Growth Rate', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)

    for ax in axes.flat:
        ax.set_xlim(0, 300)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, 'Fig8_PBR_dynamics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'saved {path}')

    print('\n' + '=' * 60 + '\nSIMULATION RESULTS SUMMARY\n' + '=' * 60)
    for key, s in summary.items():
        print(f"\n  {labels[key]}:")
        for k, v in s.items():
            print(f"    {k}: {v}")


if __name__ == '__main__':
    main()
