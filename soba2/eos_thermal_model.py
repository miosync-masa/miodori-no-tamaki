#!/usr/bin/env python3
"""
eos_thermal_model.py
====================
Canonical closed-form expressions of the EOS thermal/carbon extension
(eos_thermal/main.tex, Section 2). All figure and simulation scripts in
soba2/ import from here so that one set of equations and parameters is used.

  Eq. 6   b_env      = min(1, b_thermal, b_carbon)
  Eq. 7   beta_eff   = beta_ref(alpha) / b_env
  Eq. 8   k_r(T)     = k_r0 exp(-Ea_r / (R_g T)) / (1 + exp((T - T_m)/delta_T))
  Eq. 9   b_thermal  = k_r(T) / k_r(T_opt)            (b_thermal(T_opt) = 1)
          T_opt is defined as the temperature at which k_r(T) (Eq. 8) is maximal,
          so b_thermal <= 1 follows from the definition.
  Eq. 10  alpha_1    = 1 / ([H+]/K1 + 1 + K2/[H+])
  Eq. 11  C_i,eff    = DIC * alpha_1
  Eq. 12  b_carbon   = C_i,eff / (K_m,eff + C_i,eff)

Carbonate constants (freshwater, infinite dilution):
  pK1(T): Harned & Davis (1943)    pK1 = 3404.71/T + 0.032786 T - 14.8435
  pK2(T): Harned & Scholes (1941)  pK2 = 2902.39/T + 0.02379 T  - 6.4980
  pKw(T): Harned & Owen            pKw = 4470.99/T - 6.0875 + 0.01706 T
  K_H(T): CO2 Henry constant, K_H(298.15 K) = 0.034 mol L-1 atm-1 and
          van 't Hoff coefficient d ln K_H / d(1/T) = 2400 K (Sander 2015).

Ionic strength: the carbonate calculations use apparent (concentration) constants
at the reference ionic strength of Zarrouk medium, I_REF = 0.27 mol/L, with
activity coefficients from the Davies equation and pH on the activity scale:
  pK1' = pK1 + log g1,  pK2' = pK2 - log g1 + log g2,  pKw' = pKw + log g1,
  [H+] = 10**-pH / g1.   Pass I = 0 for the infinite-dilution constants.
The Henry constant is not corrected for salting-out.

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import numpy as np

R_GAS = 8.314e-3            # kJ mol-1 K-1
GAMMA_0 = np.cosh(1.0)**2   # Part I gate exponent
T0_K = 273.15

# Uncalibrated default thermal parameters (Ea_r 63 kJ/mol, T_m 42 C, delta_T 3 C);
# T_opt is not a free parameter but derived from them (T_opt_from).
THERMAL_DEFAULT = dict(Ea_r=63.0, T_m=42.0, dT=3.0)
KM_EFF_MM = 0.5             # effective half-saturation for HCO3- uptake (mM), CCM folded in

I_REF = 0.27                # mol/L, Zarrouk medium at its nominal recipe
KH_25 = 0.034               # mol L-1 atm-1 at 298.15 K
KH_VANT_HOFF_K = 2400.0     # K


# ══════════════════════════════════════════════════════════
# THERMAL CHANNEL (Eqs. 8-9)
# ══════════════════════════════════════════════════════════
def k_r_shape(T_C, Ea_r=63.0, T_m=42.0, dT=3.0):
    """Eq. 8 without the prefactor k_r0 (temperatures in deg C, kJ/mol)."""
    T_K = np.asarray(T_C, dtype=float) + T0_K
    return np.exp(-Ea_r / (R_GAS * T_K)) / (1.0 + np.exp((np.asarray(T_C, dtype=float) - T_m) / dT))


def T_opt_from(Ea_r=63.0, T_m=42.0, dT=3.0):
    """T_opt := temperature of maximal k_r(T) (Eq. 8), deg C. Stationary point of ln k_r:
    Ea_r / (R T^2) = s / dT with s = 1 / (1 + exp(-(T - T_m)/dT)); solved by bisection."""
    from scipy.optimize import brentq

    def g(T_C):
        T = T_C + T0_K
        return Ea_r / (R_GAS * T**2) - 1.0 / (dT * (1.0 + np.exp(-(T_C - T_m) / dT)))
    return brentq(g, T_m - 60.0 * dT - 60.0, T_m + 60.0 * dT)


def b_thermal(T_C, Ea_r=63.0, T_m=42.0, dT=3.0):
    """Eq. 9: k_r(T) / k_r(T_opt) with T_opt = argmax k_r(T); <= 1 everywhere."""
    T_opt = T_opt_from(Ea_r, T_m, dT)
    return k_r_shape(T_C, Ea_r, T_m, dT) / k_r_shape(T_opt, Ea_r, T_m, dT)


# ══════════════════════════════════════════════════════════
# CARBON CHANNEL (Eqs. 10-12)
# ══════════════════════════════════════════════════════════
def pK1(T_C):
    T = np.asarray(T_C, dtype=float) + T0_K
    return 3404.71 / T + 0.032786 * T - 14.8435


def pK2(T_C):
    T = np.asarray(T_C, dtype=float) + T0_K
    return 2902.39 / T + 0.02379 * T - 6.4980


def pKw(T_C):
    T = np.asarray(T_C, dtype=float) + T0_K
    return 4470.99 / T - 6.0875 + 0.01706 * T


def davies_log_gamma(z, I, T_C=25.0):
    """Davies equation, log10 activity coefficient of an ion of charge z.
    A(T) = 0.4883 + 8.074e-4 T_C (linear fit to the Debye-Hueckel A, 0-50 C)."""
    A = 0.4883 + 8.074e-4 * np.asarray(T_C, dtype=float)
    sI = np.sqrt(I)
    return -A * z**2 * (sI / (1.0 + sI) - 0.3 * I)


def apparent_pKs(T_C, I=I_REF):
    """(pK1', pK2', pKw', log g1) at ionic strength I (activity-scale pH)."""
    lg1, lg2 = davies_log_gamma(1, I, T_C), davies_log_gamma(2, I, T_C)
    return pK1(T_C) + lg1, pK2(T_C) - lg1 + lg2, pKw(T_C) + lg1, lg1


def K_H(T_C):
    """CO2 solubility (mol L-1 atm-1)."""
    T = np.asarray(T_C, dtype=float) + T0_K
    return KH_25 * np.exp(KH_VANT_HOFF_K * (1.0 / T - 1.0 / 298.15))


def carbonate_fractions(pH, T_C, I=I_REF):
    """(alpha_0, alpha_1, alpha_2): fractions of CO2(aq), HCO3-, CO3-- in DIC at ionic strength I."""
    H = 10.0**(-np.asarray(pH, dtype=float))         # hydrogen-ion activity
    p1, p2, _, _ = apparent_pKs(T_C, I)
    K1, K2 = 10.0**(-p1), 10.0**(-p2)
    a0 = 1.0 / (1.0 + K1 / H + K1 * K2 / H**2)
    a1 = 1.0 / (H / K1 + 1.0 + K2 / H)               # Eq. 10
    a2 = 1.0 / (H**2 / (K1 * K2) + H / K2 + 1.0)
    return a0, a1, a2


def C_i_eff(pH, DIC_mM, T_C, I=I_REF):
    """Eq. 11 (mM)."""
    return DIC_mM * carbonate_fractions(pH, T_C, I)[1]


def b_carbon(pH, DIC_mM, T_C, Km_mM=KM_EFF_MM, I=I_REF):
    """Eq. 12."""
    c = C_i_eff(pH, DIC_mM, T_C, I)
    return c / (Km_mM + c)


# ══════════════════════════════════════════════════════════
# BOTTLENECK (Eqs. 6-7) AND PI CURVE
# ══════════════════════════════════════════════════════════
def b_env(bt, bc):
    """Eq. 6: min(1, b_thermal, b_carbon)."""
    return np.minimum(1.0, np.minimum(bt, bc))


def beta_eff(beta_ref, benv):
    """Eq. 7."""
    return beta_ref / benv


def ph10_gross(I, alpha, beta, Pmax, gamma=GAMMA_0):
    """Part I double-tanh: Pmax * tanh(alpha I / Pmax) * tanh((Pmax / (beta I))**gamma)."""
    I = np.asarray(I, dtype=float)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        pcc = np.tanh(alpha * I / Pmax)
        scc = np.where(I > 0, np.tanh((Pmax / (beta * I))**gamma), 1.0)
    return Pmax * pcc * scc


def pH_from_DIC_alk(DIC_mM, alk_mM, T_C, I=I_REF):
    """Charge balance Alk = DIC (a1 + 2 a2) + [OH-] - [H+], solved by bisection on (activity) pH."""
    from scipy.optimize import brentq
    _, _, pKw_a, lg1 = apparent_pKs(T_C, I)
    DIC, Alk, Kw = DIC_mM * 1e-3, alk_mM * 1e-3, 10.0**(-pKw_a)

    def f(pH):
        aH = 10.0**(-pH)
        _, a1, a2 = carbonate_fractions(pH, T_C, I)
        return DIC * (a1 + 2 * a2) + Kw / aH - aH / 10.0**lg1 - Alk
    return brentq(f, 2.0, 14.5, xtol=1e-10)
