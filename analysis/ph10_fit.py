#!/usr/bin/env python3
"""
ph10_fit.py
===========
Ph10 model and RLC fitting protocol, as specified in the manuscript
(Main.tex Eq. 1 and Methods "SAI measurement protocol", l.319-323):

    P(I) = Pmax * tanh(alpha*I/Pmax) * tanh((Pmax/(beta*I))**gamma0) - R

  1. Acquisition : 12 actinic steps, 0-2000 umol photons m^-2 s^-1
  2. Fitting     : Ph10 with gamma0 fixed; Levenberg-Marquardt;
                   multi-start (>= 3 initial beta guesses)
  3. Quality     : accept if R^2_fit >= 0.95; otherwise reject
                   (fall back to EOS2, i.e. SAI is not measured)

Positivity of alpha, beta, Pmax is enforced by fitting their logarithms,
so the unconstrained Levenberg-Marquardt solver (method='lm') can be used.

Author: Masamichi Iizumi / Miosync, Inc.
License: MIT
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

GAMMA_0 = np.cosh(1.0)**2  # ≈ 2.381

I_RLC = np.array([0, 25, 50, 100, 150, 250, 400, 600, 900, 1200, 1600, 2000], dtype=float)

R2_MIN = 0.95                           # quality filter (Methods step 3)
S_STARTS = (1.0, 3.0, 10.0, 30.0, 100.0)  # initial beta guesses as beta0 = alpha0 / S


# ══════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════
def ph10_gross(I, alpha, beta, Pmax, gamma=GAMMA_0):
    """Offset-free Ph10: Pmax * PCC * SCC."""
    I = np.asarray(I, dtype=float)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        pcc = np.tanh(alpha * I / Pmax)
        scc = np.tanh((Pmax / (beta * I))**gamma)
        scc = np.where(I > 0, scc, 1.0)
        return Pmax * pcc * scc


def ph10_model(I, alpha, beta, Pmax, R, gamma=GAMMA_0):
    """Net Ph10: P_gross - R (Main.tex Eq. 1)."""
    return ph10_gross(I, alpha, beta, Pmax, gamma) - R


def _model_log(I, la, lb, lp, R):
    with np.errstate(over='ignore', invalid='ignore'):
        return ph10_model(I, np.exp(la), np.exp(lb), np.exp(lp), R)


# ══════════════════════════════════════════════════════════
# FITTING
# ══════════════════════════════════════════════════════════
def initial_guesses(I, P, s_starts=S_STARTS):
    """Data-derived starts: one (alpha0, Pmax0, R0) and one beta0 per S in s_starts."""
    I = np.asarray(I, dtype=float)
    P = np.asarray(P, dtype=float)
    R0 = -P[I == 0].mean() if np.any(I == 0) else 0.0
    gross = P + R0
    pmax0 = max(gross.max(), 1e-6)
    low = (I > 0) & (I <= 100)
    if not low.any():
        low = I > 0
    alpha0 = np.sum(I[low] * gross[low]) / np.sum(I[low]**2)
    if not np.isfinite(alpha0) or alpha0 <= 0:
        alpha0 = pmax0 / np.median(I[I > 0])
    return [(alpha0, alpha0 / s, pmax0, R0) for s in s_starts]


def fit_ph10_multistart(I, P, extra_starts=(), s_starts=S_STARTS, r2_min=R2_MIN):
    """
    Multi-start Levenberg-Marquardt fit of Ph10 (gamma fixed) with R^2 filter.

    Returns a dict with keys
      params    : (alpha, beta, Pmax, R) of the best start, or None if none converged
      r2        : R^2 of the best fit on the observed data
      accepted  : r2 >= r2_min
      n_starts  : number of starts tried
      n_conv    : number of starts that converged
    """
    I = np.asarray(I, dtype=float)
    P = np.asarray(P, dtype=float)
    starts = list(initial_guesses(I, P, s_starts)) + list(extra_starts)
    ss_tot = np.sum((P - P.mean())**2)

    best, best_sse, n_conv = None, np.inf, 0
    for a0, b0, pm0, r0 in starts:
        if min(a0, b0, pm0) <= 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                popt, _ = curve_fit(_model_log, I, P,
                                    p0=[np.log(a0), np.log(b0), np.log(pm0), r0],
                                    method='lm', maxfev=10000)
        except (RuntimeError, ValueError):
            continue
        resid = _model_log(I, *popt) - P
        sse = np.sum(resid**2)
        if not np.isfinite(sse):
            continue
        n_conv += 1
        if sse < best_sse:
            best_sse = sse
            best = (np.exp(popt[0]), np.exp(popt[1]), np.exp(popt[2]), popt[3])

    r2 = 1.0 - best_sse / ss_tot if (best is not None and ss_tot > 0) else -np.inf
    return {'params': best, 'r2': r2, 'accepted': bool(r2 >= r2_min),
            'n_starts': len(starts), 'n_conv': n_conv}
