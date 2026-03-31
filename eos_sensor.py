#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eos_sensor.py — Photosynthesis–Irradiance Equation of State (EOS) Soft Sensor
==============================================================================

Purpose
-------
This module implements the closed-form "soft sensor" derived from the
photosynthetic equation-of-state (EOS) framework.

It transforms a small set of measurable state variables into:
    1) a full predicted PI curve,
    2) a regime classification (R1 / R2 / R3),
    3) a stress diagnosis when beta_obs is available, and
    4) a sensor design specification via the empirical law
       NRMSE = k * sigma_SAI.

Scientific scope
----------------
The model follows the PI-curve EOS framework in which:

    P(I) = Pmax * PCC(I) * SCC(I) - R

with
    PCC(I) = tanh(alpha * I / Pmax)
    SCC(I) = tanh((Pmax / (beta * I))**gamma0)

where gamma0 = cosh^2(1) is treated as the canonical gate shape.

In the EOS framework:
    - alpha controls the light-harvesting channel (PCC),
    - beta controls the photoinhibition / stress gate (SCC),
    - S = alpha / beta is the gate variable used for regime classification.

This implementation provides two operational modes:
    - EOS2: alpha and Pmax only, beta predicted from the alpha-beta scaling law
    - EOS3: alpha, Pmax, and SAI, with beta corrected by the measured stress state

Important note on I_opt
-----------------------
This module includes a closed-form expression for the operating optimum irradiance:

    I_opt ≈ A_OPT * I_alpha^W_ALPHA * I_beta^W_BETA

where
    I_alpha = Pmax / alpha
    I_beta  = Pmax / beta

This expression is a *near-optimal operating law*, not an exact symbolic argmax
for all possible parameter ranges. In the EOS-valid domain (especially R1/R2),
it reproduces the numerically optimal operating point with small loss in
photosynthetic output. In very large-S plateau regimes, the exact location of
the global argmax becomes weakly identifiable because a broad high-irradiance
plateau develops; in such cases, the analytic law remains operationally useful
but should be interpreted as a near-optimal setpoint rather than a unique peak.

Relationship between code and paper (Eq. 12)
---------------------------------------------
The paper presents the idealized form:

    I_opt = I_alpha^{1/(gamma0+1)} * I_beta^{gamma0/(gamma0+1)}   (Eq. 12)

This code includes a numerically calibrated prefactor A_OPT ≈ 0.958 that
improves closure against the exact numerical argmax across the full S range.
The prefactor arises because Eq. 12 is derived under a power-law approximation
to the tanh-based gate balance equation; the residual from this approximation
is absorbed into A_OPT. Setting A_OPT = 1.0 recovers the paper's idealized
form with negligible additional loss (< 0.1% in Delta-P for S >= 5).

The validate_I_opt() method allows direct comparison of the closed-form
prediction against the numerical curve peak, reproducing the validation
reported in the paper's Table 7 and Supplementary Table S3.

Usage
-----
As a Python module:
    from eos_sensor import EOSSensor
    sensor = EOSSensor()
    result = sensor.predict(alpha=0.05, Pmax=8.0)

As a CLI:
    python eos_sensor.py --alpha 0.05 --Pmax 8.0
    python eos_sensor.py --alpha 0.05 --Pmax 8.0 --SAI 0.3 --R 0.5
    python eos_sensor.py --alpha 0.05 --Pmax 8.0 --beta-obs 0.01
    python eos_sensor.py --target-NRMSE 5.0 --alpha 0.05 --Pmax 8.0
    python eos_sensor.py --serve --port 5050

Author
------
M. Iizumi & T. Iizumi (Miosync, Inc.)

License
-------
MIT

Version
-------
1.1.1
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# PHYSICAL / MODEL CONSTANTS
# ============================================================================

# Canonical SCC gate shape from the EOS paper.
GAMMA_0: float = math.cosh(1.0) ** 2  # ≈ 2.3810978455418157

# Dataset-level alpha-beta scaling law:
#     log10(beta) = SCALING_M * log10(alpha) + SCALING_C
SCALING_M: float = 0.814
SCALING_C: float = -1.355

# Empirical design law from the EOS sensing paper:
#     NRMSE (%) = DESIGN_K * sigma_SAI
DESIGN_K: float = 50.4

# Regime boundaries in gate variable S = alpha / beta
S_BOUNDARY_R1R2: float = 10.0
S_BOUNDARY_R2R3: float = 3.0

# Forbidden zone observed in the dataset (structural depletion zone)
FZ_LOW: float = 0.82
FZ_HIGH: float = 1.61

# ---------------------------------------------------------------------------
# Closed-form near-optimal irradiance law
# ---------------------------------------------------------------------------
# Paper Eq. 12 (idealized):
#     I_opt = I_alpha^{1/(gamma0+1)} * I_beta^{gamma0/(gamma0+1)}
#
# Implementation (with numerical closure prefactor):
#     I_opt = A_OPT * I_alpha^W_ALPHA * I_beta^W_BETA
#
# The weights W_ALPHA and W_BETA are determined solely by gamma0 and
# correspond to a weighted geometric mean biased 70.4% toward the
# inhibition scale I_beta. The prefactor A_OPT ≈ 0.958 absorbs the
# residual from the power-law approximation used to derive Eq. 12.
# Setting A_OPT = 1.0 recovers the paper's idealized form.
# ---------------------------------------------------------------------------
W_BETA: float = GAMMA_0 / (GAMMA_0 + 1.0)   # ≈ 0.7042 weight on I_beta
W_ALPHA: float = 1.0 / (GAMMA_0 + 1.0)      # ≈ 0.2958 weight on I_alpha
A_OPT: float = 0.95808                      # prefactor from numeric closure

# Useful threshold above which the PI curve often enters a broad plateau regime.
# In such cases, "the" exact argmax location is weakly identifiable.
LARGE_S_PLATEAU_THRESHOLD: float = 50.0


# ============================================================================
# CORE SCALING / REGIME FUNCTIONS
# ============================================================================

def beta_pred(alpha: float) -> float:
    """
    Predict beta from alpha using the dataset-level scaling law.

    Scaling law
    -----------
    log10(beta) = 0.814 * log10(alpha) - 1.355

    Parameters
    ----------
    alpha : float
        Light-harvesting efficiency (initial slope of the PI curve).
        Must be strictly positive.

    Returns
    -------
    float
        Predicted beta value.

    Raises
    ------
    ValueError
        If alpha <= 0.

    Notes
    -----
    This is the population-level baseline relation used by EOS2.
    In EOS3, the effective beta is adjusted by SAI:
        beta_eff = beta_pred(alpha) * 10**SAI
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return 10 ** (SCALING_M * math.log10(alpha) + SCALING_C)


def classify_regime(S: float) -> Dict[str, Any]:
    """
    Classify the photosynthetic regime from the gate variable S = alpha / beta.

    Regimes
    -------
    R1 : S > 10
         Factorized regime. PCC and SCC are well separated.
         EOS2 is usually sufficient.

    R2 : 3 < S <= 10
         Transition regime. SCC begins to affect the plateau region.
         EOS3 is recommended when SAI is available.

    R3 : S <= 3
         Coupled regime. Outside the intended EOS validity domain.

    Parameters
    ----------
    S : float
        Gate variable alpha / beta.

    Returns
    -------
    dict
        Dictionary with regime code, human-readable label, EOS tier suggestion,
        and forbidden-zone membership.
    """
    in_fz = FZ_LOW < S < FZ_HIGH

    if S > S_BOUNDARY_R1R2:
        return {
            "regime": "R1",
            "label": "Factorized — PCC/SCC well separated",
            "eos_tier": "EOS2",
            "S": round(S, 4),
            "in_forbidden_zone": in_fz,
        }
    elif S > S_BOUNDARY_R2R3:
        return {
            "regime": "R2",
            "label": "Transition — SCC affects the plateau",
            "eos_tier": "EOS3",
            "S": round(S, 4),
            "in_forbidden_zone": in_fz,
        }
    else:
        return {
            "regime": "R3",
            "label": "Coupled — outside EOS validity",
            "eos_tier": "NONE",
            "S": round(S, 4),
            "in_forbidden_zone": in_fz,
        }


# ============================================================================
# PI-CURVE BUILDING BLOCKS
# ============================================================================

def _safe_tanh_power(x: float, gamma: float) -> float:
    """
    Safely evaluate tanh(x**gamma) with overflow protection.

    Parameters
    ----------
    x : float
        Positive scalar argument before exponentiation.
    gamma : float
        Positive exponent.

    Returns
    -------
    float
        Value in [0, 1].

    Notes
    -----
    For large x**gamma, tanh(.) rapidly saturates to 1.
    We cap the internal argument at a practical threshold to avoid needless
    floating-point overflow while preserving physical meaning.
    """
    if x <= 0:
        return 0.0

    try:
        val = x ** gamma
    except OverflowError:
        return 1.0

    # tanh(20) is already essentially 1 at double precision.
    if val > 20.0:
        return 1.0

    return math.tanh(val)


def pi_curve(
    I: float,
    Pmax: float,
    alpha: float,
    beta: float,
    gamma: float = GAMMA_0,
    R: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate the PI model at a single irradiance I.

    Model
    -----
    P_gross(I) = Pmax * PCC(I) * SCC(I)
    P_net(I)   = P_gross(I) - R

    with
        PCC(I) = tanh(alpha * I / Pmax)
        SCC(I) = tanh((Pmax / (beta * I))**gamma)

    Parameters
    ----------
    I : float
        Irradiance (typically µmol photons m^-2 s^-1).
    Pmax : float
        Maximum photosynthetic rate.
    alpha : float
        Light-harvesting efficiency.
    beta : float
        Photoinhibition susceptibility.
    gamma : float, optional
        SCC gate shape parameter. Default is GAMMA_0.
    R : float, optional
        Dark offset / respiration-like offset.

    Returns
    -------
    dict
        Dictionary containing:
        - I
        - P_gross
        - P_net
        - PCC
        - SCC

    Raises
    ------
    ValueError
        If Pmax <= 0, alpha <= 0, beta <= 0, or gamma <= 0.

    Notes
    -----
    For I <= 0, the model returns:
        PCC = 0, SCC = 1, P_gross = 0, P_net = -R
    which is the natural extension of the formula to the dark point.
    """
    if Pmax <= 0:
        raise ValueError(f"Pmax must be positive, got {Pmax}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    if I <= 0:
        return {"I": I, "P_gross": 0.0, "P_net": -R, "PCC": 0.0, "SCC": 1.0}

    pcc = math.tanh(alpha * I / Pmax)
    scc = _safe_tanh_power(Pmax / (beta * I), gamma)
    p_gross = Pmax * pcc * scc
    p_net = p_gross - R

    return {
        "I": round(I, 6),
        "P_gross": round(p_gross, 6),
        "P_net": round(p_net, 6),
        "PCC": round(pcc, 6),
        "SCC": round(scc, 6),
    }


def I_opt_analytic(alpha: float, Pmax: float, beta: float) -> float:
    """
    Closed-form near-optimal irradiance for the Ph10 / EOS architecture.

    Formula (implementation)
    ------------------------
    I_opt ≈ A_OPT * I_alpha^W_ALPHA * I_beta^W_BETA

    where
        I_alpha = Pmax / alpha
        I_beta  = Pmax / beta
        W_ALPHA = 1 / (gamma0 + 1)     ≈ 0.2958
        W_BETA  = gamma0 / (gamma0 + 1) ≈ 0.7042
        A_OPT   ≈ 0.958                (numerical closure prefactor)

    Paper form (Eq. 12)
    -------------------
    I_opt = I_alpha^{1/(gamma0+1)} * I_beta^{gamma0/(gamma0+1)}

    The paper presents the idealized geometric mean without the prefactor.
    A_OPT absorbs the residual from the power-law approximation used in
    deriving Eq. 12 from the exact dP/dI = 0 condition. Setting A_OPT = 1.0
    recovers the paper form; the additional loss is < 0.1% in Delta-P for
    S >= 5.

    Equivalent interpretation
    -------------------------
    This is a weighted geometric mean of the PCC saturation scale and the SCC
    onset scale, with the weight biased toward the inhibition scale. The
    weights are determined solely by gamma0 = cosh^2(1).

    Parameters
    ----------
    alpha : float
        Light-harvesting efficiency.
    Pmax : float
        Maximum photosynthetic rate.
    beta : float
        Photoinhibition susceptibility.

    Returns
    -------
    float
        Closed-form near-optimal irradiance.

    Raises
    ------
    ValueError
        If alpha <= 0, Pmax <= 0, or beta <= 0.

    Important caveat
    ----------------
    In very large-S plateau regimes, the exact peak location becomes weakly
    identifiable because P(I) is nearly flat near Pmax. The returned value
    should be interpreted as an operational setpoint that preserves near-maximal
    photosynthesis, rather than a unique mathematically privileged maximizer.
    """
    if alpha <= 0 or Pmax <= 0 or beta <= 0:
        raise ValueError("alpha, Pmax, and beta must all be positive")

    i_alpha = Pmax / alpha
    i_beta = Pmax / beta
    return A_OPT * (i_alpha ** W_ALPHA) * (i_beta ** W_BETA)


def I_opt_paper(alpha: float, Pmax: float, beta: float) -> float:
    """
    Idealized closed-form optimal irradiance as written in the paper (Eq. 12).

    Formula
    -------
    I_opt = I_alpha^{1/(gamma0+1)} * I_beta^{gamma0/(gamma0+1)}

    This is the A_OPT = 1.0 form. Use I_opt_analytic() for the numerically
    calibrated version.

    Parameters
    ----------
    alpha : float
        Light-harvesting efficiency.
    Pmax : float
        Maximum photosynthetic rate.
    beta : float
        Photoinhibition susceptibility.

    Returns
    -------
    float
        Idealized optimal irradiance (no prefactor).
    """
    if alpha <= 0 or Pmax <= 0 or beta <= 0:
        raise ValueError("alpha, Pmax, and beta must all be positive")

    i_alpha = Pmax / alpha
    i_beta = Pmax / beta
    return (i_alpha ** W_ALPHA) * (i_beta ** W_BETA)


# ============================================================================
# RESULT CONTAINER
# ============================================================================

@dataclass
class EOSResult:
    """
    Container for a complete EOS prediction.

    Fields
    ------
    Inputs
        alpha, Pmax, SAI, R

    Derived parameters
        beta_predicted
        beta_effective
        S

    Regime / validity
        regime
        regime_label
        eos_tier
        in_forbidden_zone

    Diagnostics
        expected_NRMSE_pct
        sigma_SAI
        I_alpha
        I_beta
        I_opt_closed_form   : near-optimal irradiance (with A_OPT prefactor)
        I_opt_paper_form    : idealized form from Eq. 12 (A_OPT = 1.0)
        I_opt_curve_peak    : numerical peak from sampled PI curve

    Notes
        Human-readable warnings / caveats intended for operators and reviewers.

    Curve
        A list of PI-curve samples, each with P_gross, P_net, PCC, and SCC.
    """

    # Input echo
    alpha: float
    Pmax: float
    SAI: Optional[float]
    R: float

    # Derived parameters
    beta_predicted: float
    beta_effective: float
    S: float

    # Regime
    regime: str
    regime_label: str
    eos_tier: str
    in_forbidden_zone: bool

    # Quality / uncertainty
    expected_NRMSE_pct: Optional[float]
    sigma_SAI: Optional[float]

    # Diagnostics
    I_alpha: float
    I_beta: float
    I_opt_closed_form: float
    I_opt_paper_form: float
    I_opt_curve_peak: float

    # Optional human-readable notes
    notes: List[str] = field(default_factory=list)

    # PI-curve samples
    curve: List[Dict[str, float]] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the result as JSON."""
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


# ============================================================================
# VALIDATION RESULT CONTAINER
# ============================================================================

@dataclass
class IoptValidationResult:
    """
    Result of validate_I_opt(): comparison of closed-form vs numerical optimum.

    This dataclass reproduces the validation reported in the paper's Table 7
    and Supplementary Table S3.

    Fields
    ------
    S : float
        Gate variable alpha / beta.
    regime : str
        R1 / R2 / R3.
    I_opt_numeric : float
        Numerically determined peak irradiance (fine-grid argmax of P_gross).
    I_opt_closed_form : float
        I_opt from I_opt_analytic() (with A_OPT prefactor).
    I_opt_paper_form : float
        I_opt from I_opt_paper() (Eq. 12, no prefactor).
    P_gross_at_numeric : float
        P_gross evaluated at the numerical optimum.
    P_gross_at_closed_form : float
        P_gross evaluated at the closed-form optimum.
    P_gross_at_paper_form : float
        P_gross evaluated at the paper-form optimum.
    delta_I_closed_form_pct : float
        Location error (%) of closed-form vs numerical optimum.
    delta_I_paper_form_pct : float
        Location error (%) of paper-form vs numerical optimum.
    delta_P_closed_form_pct : float
        Photosynthetic loss (%) at the closed-form optimum.
    delta_P_paper_form_pct : float
        Photosynthetic loss (%) at the paper-form optimum.
    plateau_warning : bool
        True if S >= LARGE_S_PLATEAU_THRESHOLD (broad plateau regime).
    """
    S: float
    regime: str
    I_opt_numeric: float
    I_opt_closed_form: float
    I_opt_paper_form: float
    P_gross_at_numeric: float
    P_gross_at_closed_form: float
    P_gross_at_paper_form: float
    delta_I_closed_form_pct: float
    delta_I_paper_form_pct: float
    delta_P_closed_form_pct: float
    delta_P_paper_form_pct: float
    plateau_warning: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MAIN SENSOR CLASS
# ============================================================================

class EOSSensor:
    """
    Photosynthesis–Irradiance Equation of State Soft Sensor.

    The sensor predicts PI behavior from a minimal set of inputs.

    Modes
    -----
    EOS2:
        Inputs: alpha, Pmax
        beta is predicted from the alpha-beta scaling law.

    EOS3:
        Inputs: alpha, Pmax, SAI
        beta is corrected by the measured stress adaptation state.

    Methods
    -------
    predict()
        Generate a full PI curve prediction from state variables.
    diagnose()
        Classify a measured curve state when observed beta is available.
    design_spec()
        Convert a desired prediction accuracy into a required sigma_SAI.
    validate_I_opt()
        Compare closed-form I_opt against numerical PI curve peak.
        Reproduces the validation in paper Table 7 / SI Table S3.

    Notes
    -----
    - Regime classification is based on S = alpha / beta_effective.
    - The forbidden zone flag is structural: it indicates that S lies in the
      empirically depleted interval observed in the EOS dataset.
    - R3 is returned for completeness, but it lies outside the intended
      factorized EOS validity domain.
    """

    def __init__(
        self,
        I_range: tuple[float, float] = (1.0, 2500.0),
        n_points: int = 200,
        gamma: float = GAMMA_0,
    ) -> None:
        """
        Parameters
        ----------
        I_range : tuple, optional
            Irradiance range used to generate the predicted PI curve.
        n_points : int, optional
            Number of sampled points in the generated curve.
        gamma : float, optional
            SCC gate shape parameter. Defaults to GAMMA_0.
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        if I_range[0] < 0 or I_range[1] <= I_range[0]:
            raise ValueError("I_range must satisfy 0 <= I_min < I_max")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.I_min, self.I_max = I_range
        self.n_points = n_points
        self.gamma = gamma

    def predict(
        self,
        alpha: float,
        Pmax: float,
        SAI: Optional[float] = None,
        R: float = 0.0,
        sigma_SAI: Optional[float] = None,
        I_values: Optional[List[float]] = None,
    ) -> EOSResult:
        """
        Predict a PI curve and associated diagnostics from state variables.

        Parameters
        ----------
        alpha : float
            Light-harvesting efficiency.
        Pmax : float
            Maximum photosynthetic rate.
        SAI : float, optional
            Stress Adaptation Index. If omitted, EOS2 is used.
        R : float, optional
            Dark offset / respiration-like offset.
        sigma_SAI : float, optional
            Measurement uncertainty of SAI. If provided, the expected NRMSE
            is estimated from NRMSE = DESIGN_K * sigma_SAI.
        I_values : list of float, optional
            Custom irradiance grid. If omitted, a uniform grid is generated.

        Returns
        -------
        EOSResult
            Full prediction package.

        Raises
        ------
        ValueError
            If alpha <= 0 or Pmax <= 0.

        Notes
        -----
        - In EOS2, beta_effective = beta_pred(alpha).
        - In EOS3, beta_effective = beta_pred(alpha) * 10**SAI.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if Pmax <= 0:
            raise ValueError(f"Pmax must be > 0, got {Pmax}")

        notes: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: baseline beta from the scaling law
        # ------------------------------------------------------------------
        bp = beta_pred(alpha)

        # ------------------------------------------------------------------
        # Step 2: effective beta after optional SAI correction
        # ------------------------------------------------------------------
        if SAI is not None:
            beta_eff = bp * (10 ** SAI)
            eos_mode = "EOS3"
        else:
            beta_eff = bp
            eos_mode = "EOS2"

        # ------------------------------------------------------------------
        # Step 3: regime classification from S = alpha / beta_eff
        # ------------------------------------------------------------------
        S = alpha / beta_eff
        regime_info = classify_regime(S)

        if regime_info["in_forbidden_zone"]:
            notes.append(
                "S lies inside the empirically depleted forbidden zone; "
                "interpret regime assignment with caution."
            )

        if regime_info["regime"] == "R3":
            notes.append(
                "R3 corresponds to the coupled regime and lies outside the "
                "intended factorized EOS validity domain."
            )

        if S >= LARGE_S_PLATEAU_THRESHOLD:
            notes.append(
                "Large-S plateau regime detected: exact argmax location may be "
                "weakly identifiable; I_opt_closed_form should be interpreted "
                "as a near-optimal operating setpoint."
            )

        # ------------------------------------------------------------------
        # Step 4: characteristic irradiances
        # ------------------------------------------------------------------
        I_alpha = Pmax / alpha
        I_beta = Pmax / beta_eff

        # ------------------------------------------------------------------
        # Step 5: build irradiance grid and generate PI curve
        # ------------------------------------------------------------------
        if I_values is None:
            step = (self.I_max - self.I_min) / (self.n_points - 1)
            I_values = [self.I_min + i * step for i in range(self.n_points)]

        curve = [
            pi_curve(I, Pmax, alpha, beta_eff, self.gamma, R)
            for I in I_values
        ]

        # Sampled curve peak (useful sanity check, not the symbolic optimum)
        peak = max(curve, key=lambda p: p["P_gross"])
        I_opt_curve_peak = float(peak["I"])

        # Closed-form near-optimal operating point (with and without prefactor)
        i_opt_cf = I_opt_analytic(alpha, Pmax, beta_eff)
        i_opt_paper = I_opt_paper(alpha, Pmax, beta_eff)

        # ------------------------------------------------------------------
        # Step 6: error estimate from sigma_SAI if available
        # ------------------------------------------------------------------
        expected_nrmse_pct: Optional[float]
        if sigma_SAI is not None:
            expected_nrmse_pct = DESIGN_K * sigma_SAI
        else:
            expected_nrmse_pct = None

        return EOSResult(
            alpha=round(alpha, 6),
            Pmax=round(Pmax, 6),
            SAI=round(SAI, 6) if SAI is not None else None,
            R=round(R, 6),
            beta_predicted=round(bp, 6),
            beta_effective=round(beta_eff, 6),
            S=round(S, 4),
            regime=regime_info["regime"],
            regime_label=regime_info["label"],
            eos_tier=eos_mode,
            in_forbidden_zone=regime_info["in_forbidden_zone"],
            expected_NRMSE_pct=round(expected_nrmse_pct, 3) if expected_nrmse_pct is not None else None,
            sigma_SAI=round(sigma_SAI, 6) if sigma_SAI is not None else None,
            I_alpha=round(I_alpha, 3),
            I_beta=round(I_beta, 3),
            I_opt_closed_form=round(i_opt_cf, 3),
            I_opt_paper_form=round(i_opt_paper, 3),
            I_opt_curve_peak=round(I_opt_curve_peak, 3),
            notes=notes,
            curve=curve,
        )

    def validate_I_opt(
        self,
        alpha: float,
        Pmax: float,
        beta: float,
        n_fine: int = 10000,
        I_max: Optional[float] = None,
    ) -> IoptValidationResult:
        """
        Compare the closed-form I_opt against the numerical PI-curve peak.

        This method reproduces the validation reported in the paper's Table 7
        and Supplementary Table S3. It evaluates the PI curve on a fine
        irradiance grid, locates the numerical argmax of P_gross, and computes
        location error (Delta-I) and photosynthetic loss (Delta-P) for both
        the A_OPT-calibrated form and the paper's idealized form (Eq. 12).

        Parameters
        ----------
        alpha : float
            Light-harvesting efficiency.
        Pmax : float
            Maximum photosynthetic rate.
        beta : float
            Photoinhibition susceptibility (observed or predicted).
        n_fine : int, optional
            Number of irradiance points for the fine-grid search.
            Default: 10000 (sufficient for sub-0.1% precision).
        I_max : float, optional
            Upper bound of the irradiance search grid. If None, automatically
            set to max(5 * Pmax/beta, 3 * Pmax/alpha) to ensure the peak
            is captured even in plateau regimes.

        Returns
        -------
        IoptValidationResult
            Dataclass with numerical vs closed-form comparison metrics.

        Raises
        ------
        ValueError
            If alpha, Pmax, or beta are not strictly positive.

        Examples
        --------
        >>> sensor = EOSSensor()
        >>> v = sensor.validate_I_opt(alpha=0.05, Pmax=10.0, beta=0.01)
        >>> print(f"S={v.S:.1f}, Delta-P(closed)={v.delta_P_closed_form_pct:.4f}%")
        S=5.0, Delta-P(closed)=0.0312%

        >>> # Sweep across S values (reproduces Table 7)
        >>> for S_target in [5, 7, 10, 15, 20, 50]:
        ...     beta_val = 0.05 / S_target
        ...     v = sensor.validate_I_opt(alpha=0.05, Pmax=10.0, beta=beta_val)
        ...     print(f"S={v.S:5.1f}  {v.regime}  "
        ...           f"Delta-I={v.delta_I_paper_form_pct:5.1f}%  "
        ...           f"Delta-P={v.delta_P_paper_form_pct:.4f}%")

        Notes
        -----
        In the large-S plateau regime (S >= 50), the numerical peak location
        becomes ambiguous because P_gross is nearly flat. Delta-I may be large
        but Delta-P remains near zero — this is the expected physical behavior
        documented in SI Section S5.1.
        """
        if alpha <= 0 or Pmax <= 0 or beta <= 0:
            raise ValueError("alpha, Pmax, and beta must all be positive")

        S = alpha / beta
        regime_info = classify_regime(S)

        # Determine search range
        I_alpha = Pmax / alpha
        I_beta = Pmax / beta
        if I_max is None:
            I_max = max(5.0 * I_beta, 3.0 * I_alpha, 5000.0)

        # Fine-grid numerical search for argmax(P_gross)
        I_step = I_max / n_fine
        best_I = 0.0
        best_P = -1.0
        for k in range(1, n_fine + 1):
            I_val = k * I_step
            pt = pi_curve(I_val, Pmax, alpha, beta, self.gamma, 0.0)
            if pt["P_gross"] > best_P:
                best_P = pt["P_gross"]
                best_I = I_val

        I_num = best_I
        P_num = best_P

        # Closed-form predictions
        I_cf = I_opt_analytic(alpha, Pmax, beta)
        I_pf = I_opt_paper(alpha, Pmax, beta)

        # Evaluate P_gross at closed-form operating points
        P_cf = pi_curve(I_cf, Pmax, alpha, beta, self.gamma, 0.0)["P_gross"]
        P_pf = pi_curve(I_pf, Pmax, alpha, beta, self.gamma, 0.0)["P_gross"]

        # Error metrics
        delta_I_cf = abs(I_cf - I_num) / I_num * 100.0 if I_num > 0 else 0.0
        delta_I_pf = abs(I_pf - I_num) / I_num * 100.0 if I_num > 0 else 0.0
        delta_P_cf = (P_num - P_cf) / P_num * 100.0 if P_num > 0 else 0.0
        delta_P_pf = (P_num - P_pf) / P_num * 100.0 if P_num > 0 else 0.0

        return IoptValidationResult(
            S=round(S, 4),
            regime=regime_info["regime"],
            I_opt_numeric=round(I_num, 3),
            I_opt_closed_form=round(I_cf, 3),
            I_opt_paper_form=round(I_pf, 3),
            P_gross_at_numeric=round(P_num, 6),
            P_gross_at_closed_form=round(P_cf, 6),
            P_gross_at_paper_form=round(P_pf, 6),
            delta_I_closed_form_pct=round(delta_I_cf, 3),
            delta_I_paper_form_pct=round(delta_I_pf, 3),
            delta_P_closed_form_pct=round(max(delta_P_cf, 0.0), 6),
            delta_P_paper_form_pct=round(max(delta_P_pf, 0.0), 6),
            plateau_warning=S >= LARGE_S_PLATEAU_THRESHOLD,
        )

    def diagnose(self, alpha: float, Pmax: float, beta_obs: float) -> Dict[str, Any]:
        """
        Diagnose a measured curve state when observed beta is available.

        Parameters
        ----------
        alpha : float
            Observed alpha.
        Pmax : float
            Observed Pmax.
        beta_obs : float
            Observed beta from a fitted PI curve.

        Returns
        -------
        dict
            Diagnostic summary including:
            - predicted beta
            - observed beta
            - SAI
            - regime classification
            - operator-facing recommendations
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if Pmax <= 0:
            raise ValueError(f"Pmax must be > 0, got {Pmax}")
        if beta_obs <= 0:
            raise ValueError(f"beta_obs must be > 0, got {beta_obs}")

        bp = beta_pred(alpha)
        sai = math.log10(beta_obs) - math.log10(bp)
        S = alpha / beta_obs
        regime_info = classify_regime(S)

        recommendations: List[str] = []

        if regime_info["regime"] == "R1":
            recommendations.append(
                "EOS2 is typically sufficient in R1. SAI may be tracked diagnostically but is usually not required for prediction."
            )
        elif regime_info["regime"] == "R2":
            recommendations.append(
                "EOS3 is recommended in R2. Include SAI for accurate prediction near the transition regime."
            )
            if sai > 0.5:
                recommendations.append(
                    "High positive SAI: elevated photoinhibition susceptibility relative to baseline. Consider reducing irradiance or checking environmental bottlenecks."
                )
            elif sai < -0.5:
                recommendations.append(
                    "Negative SAI: stronger photoprotection / acclimation than baseline."
                )
        else:
            recommendations.append(
                "R3 detected: the factorized EOS is not intended as a quantitative predictive model in this regime."
            )

        if regime_info["in_forbidden_zone"]:
            recommendations.append(
                "Forbidden-zone proximity detected. Treat the state as structurally unusual and monitor carefully."
            )

        return {
            "alpha": round(alpha, 6),
            "Pmax": round(Pmax, 6),
            "beta_predicted": round(bp, 6),
            "beta_observed": round(beta_obs, 6),
            "SAI": round(sai, 6),
            **regime_info,
            "recommendations": recommendations,
        }

    def design_spec(self, target_NRMSE_pct: float) -> Dict[str, Any]:
        """
        Convert a desired prediction accuracy into a required sigma_SAI.

        Parameters
        ----------
        target_NRMSE_pct : float
            Desired NRMSE in percent.

        Returns
        -------
        dict
            Sensor design requirement based on the empirical design law.
        """
        if target_NRMSE_pct <= 0:
            raise ValueError("target_NRMSE_pct must be positive")

        sigma_req = target_NRMSE_pct / DESIGN_K
        return {
            "target_NRMSE_pct": round(target_NRMSE_pct, 4),
            "required_sigma_SAI": round(sigma_req, 6),
            "design_law": f"NRMSE = {DESIGN_K} * sigma_SAI",
            "note": (
                f"To achieve NRMSE < {target_NRMSE_pct:.3f}%, "
                f"the SAI measurement uncertainty should be < {sigma_req:.3f}. "
                "This requirement follows the dataset-anchored EOS design law."
            ),
        }


# ============================================================================
# FLASK API (OPTIONAL)
# ============================================================================

def serve(host: str = "0.0.0.0", port: int = 5050) -> None:
    """
    Start a minimal Flask API for the EOS soft sensor.

    Endpoints
    ---------
    POST /predict
        JSON body:
            {"alpha": 0.05, "Pmax": 8.0, "SAI": 0.2, "R": 0.0, "sigma_SAI": 0.1}

    POST /diagnose
        JSON body:
            {"alpha": 0.05, "Pmax": 8.0, "beta_obs": 0.01}

    POST /design
        JSON body:
            {"target_NRMSE_pct": 5.0}

    POST /validate_iopt
        JSON body:
            {"alpha": 0.05, "Pmax": 10.0, "beta": 0.01}

    GET /health
        Returns model constants and version info.

    Notes
    -----
    Flask is optional. This function imports Flask lazily so that the module
    remains usable without Flask installed.
    """
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return

    app = Flask(__name__)
    sensor = EOSSensor(n_points=100)

    @app.route("/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(force=True)
        try:
            result = sensor.predict(
                alpha=data["alpha"],
                Pmax=data["Pmax"],
                SAI=data.get("SAI"),
                R=data.get("R", 0.0),
                sigma_SAI=data.get("sigma_SAI"),
            )
            return jsonify(asdict(result))
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/diagnose", methods=["POST"])
    def api_diagnose():
        data = request.get_json(force=True)
        try:
            result = sensor.diagnose(
                alpha=data["alpha"],
                Pmax=data["Pmax"],
                beta_obs=data["beta_obs"],
            )
            return jsonify(result)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/design", methods=["POST"])
    def api_design():
        data = request.get_json(force=True)
        try:
            result = sensor.design_spec(data["target_NRMSE_pct"])
            return jsonify(result)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/validate_iopt", methods=["POST"])
    def api_validate_iopt():
        data = request.get_json(force=True)
        try:
            result = sensor.validate_I_opt(
                alpha=data["alpha"],
                Pmax=data["Pmax"],
                beta=data["beta"],
                n_fine=data.get("n_fine", 10000),
            )
            return jsonify(result.to_dict())
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "version": "1.1.1",
            "constants": {
                "gamma_0": GAMMA_0,
                "scaling_m": SCALING_M,
                "scaling_c": SCALING_C,
                "design_k": DESIGN_K,
                "w_alpha": W_ALPHA,
                "w_beta": W_BETA,
                "A_opt": A_OPT,
                "S_boundary_R1R2": S_BOUNDARY_R1R2,
                "S_boundary_R2R3": S_BOUNDARY_R2R3,
                "FZ_low": FZ_LOW,
                "FZ_high": FZ_HIGH,
            }
        })

    print(f"EOS Soft Sensor API running on {host}:{port}")
    print("Endpoints:")
    print("  POST /predict")
    print("  POST /diagnose")
    print("  POST /design")
    print("  POST /validate_iopt")
    print("  GET  /health")
    app.run(host=host, port=port)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Command-line interface for the EOS soft sensor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EOS Soft Sensor — PI curve prediction from minimal inputs",
        epilog=(
            "Examples:\n"
            "  python eos_sensor.py --alpha 0.05 --Pmax 8.0\n"
            "  python eos_sensor.py --alpha 0.05 --Pmax 8.0 --SAI 0.2\n"
            "  python eos_sensor.py --alpha 0.05 --Pmax 8.0 --beta-obs 0.01\n"
            "  python eos_sensor.py --target-NRMSE 5.0 --alpha 0.05 --Pmax 8.0\n"
            "  python eos_sensor.py --validate-iopt --alpha 0.05 --Pmax 10.0 --beta-obs 0.01\n"
            "  python eos_sensor.py --serve --port 5050"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--alpha", type=float, default=None, help="Light-harvesting efficiency")
    parser.add_argument("--Pmax", type=float, default=None, help="Maximum photosynthetic rate")
    parser.add_argument("--SAI", type=float, default=None, help="Stress Adaptation Index (optional)")
    parser.add_argument("--R", type=float, default=0.0, help="Dark offset / respiration-like offset")
    parser.add_argument("--sigma-SAI", dest="sigma_SAI", type=float, default=None,
                        help="Measurement uncertainty in SAI for expected NRMSE estimation")
    parser.add_argument("--beta-obs", dest="beta_obs", type=float, default=None,
                        help="Observed beta for diagnosis mode or validate-iopt mode")
    parser.add_argument("--target-NRMSE", dest="target_NRMSE", type=float, default=None,
                        help="Target NRMSE in percent for design-spec mode")
    parser.add_argument("--validate-iopt", dest="validate_iopt", action="store_true",
                        help="Validate closed-form I_opt against numerical peak (requires --alpha, --Pmax, --beta-obs)")
    parser.add_argument("--json", action="store_true", help="Output full JSON")
    parser.add_argument("--compact", action="store_true", help="Suppress curve samples in JSON output")
    parser.add_argument("--n-points", dest="n_points", type=int, default=50,
                        help="Number of irradiance points in CLI prediction output")
    parser.add_argument("--serve", action="store_true", help="Start Flask API server")
    parser.add_argument("--port", type=int, default=5050, help="API server port (default: 5050)")

    args = parser.parse_args()

    # Server mode does not require alpha / Pmax.
    if args.serve:
        serve(port=args.port)
        return

    # All other modes require alpha and Pmax.
    if args.alpha is None or args.Pmax is None:
        parser.error("--alpha and --Pmax are required unless using --serve")

    sensor = EOSSensor(n_points=args.n_points)

    # ----------------------------------------------------------------------
    # Validate I_opt mode
    # ----------------------------------------------------------------------
    if args.validate_iopt:
        if args.beta_obs is None:
            parser.error("--validate-iopt requires --beta-obs")
        result = sensor.validate_I_opt(args.alpha, args.Pmax, args.beta_obs)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    # ----------------------------------------------------------------------
    # Design-spec mode
    # ----------------------------------------------------------------------
    if args.target_NRMSE is not None:
        spec = sensor.design_spec(args.target_NRMSE)
        print(json.dumps(spec, indent=2, ensure_ascii=False))
        return

    # ----------------------------------------------------------------------
    # Diagnosis mode
    # ----------------------------------------------------------------------
    if args.beta_obs is not None:
        diag = sensor.diagnose(args.alpha, args.Pmax, args.beta_obs)
        print(json.dumps(diag, indent=2, ensure_ascii=False))
        return

    # ----------------------------------------------------------------------
    # Prediction mode
    # ----------------------------------------------------------------------
    result = sensor.predict(
        alpha=args.alpha,
        Pmax=args.Pmax,
        SAI=args.SAI,
        R=args.R,
        sigma_SAI=args.sigma_SAI,
    )

    if args.json:
        data = asdict(result)
        if args.compact:
            data.pop("curve", None)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Human-readable output
    print("=" * 72)
    print("EOS SOFT SENSOR — PI Curve Prediction")
    print("=" * 72)
    print(f"Mode                  : {result.eos_tier}")
    print(f"Regime                : {result.regime} ({result.regime_label})")
    print(f"S = alpha / beta      : {result.S}")
    print(f"In forbidden zone     : {result.in_forbidden_zone}")
    print("-" * 72)
    print(f"alpha                 : {result.alpha}")
    print(f"Pmax                  : {result.Pmax}")
    print(f"SAI                   : {result.SAI}")
    print(f"R                     : {result.R}")
    print("-" * 72)
    print(f"beta_predicted        : {result.beta_predicted}")
    print(f"beta_effective        : {result.beta_effective}")
    print(f"I_alpha               : {result.I_alpha}")
    print(f"I_beta                : {result.I_beta}")
    print(f"I_opt (closed-form)   : {result.I_opt_closed_form}")
    print(f"I_opt (paper Eq.12)   : {result.I_opt_paper_form}")
    print(f"I_opt (curve peak)    : {result.I_opt_curve_peak}")
    print(f"Expected NRMSE (%)    : {result.expected_NRMSE_pct}")
    if result.notes:
        print("-" * 72)
        print("Notes:")
        for i, note in enumerate(result.notes, start=1):
            print(f"  {i}. {note}")
    print("=" * 72)

    # Print sampled curve summary
    print(f"\n{'I':>10} {'P_gross':>12} {'P_net':>12} {'PCC':>10} {'SCC':>10}")
    print("-" * 60)
    step = max(1, len(result.curve) // 20)
    for idx in range(0, len(result.curve), step):
        p = result.curve[idx]
        print(
            f"{p['I']:10.2f}"
            f"{p['P_gross']:12.4f}"
            f"{p['P_net']:12.4f}"
            f"{p['PCC']:10.4f}"
            f"{p['SCC']:10.4f}"
        )


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
