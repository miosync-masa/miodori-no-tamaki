#!/usr/bin/env python3
"""
eos_sensor.py — Photosynthesis–Irradiance Equation of State (EOS) Soft Sensor
==============================================================================

Predicts complete PI curves from minimal PAM/fluorometer measurements.
Based on: Iizumi (2026) "Equation of state for photosynthesis–irradiance
curves", Biochemical Engineering Journal (submitted).

Usage:
------
    # As Python module
    from eos_sensor import EOSSensor
    sensor = EOSSensor()
    result = sensor.predict(alpha=0.05, Pmax=8.0)

    # As CLI
    python eos_sensor.py --alpha 0.05 --Pmax 8.0
    python eos_sensor.py --alpha 0.05 --Pmax 8.0 --SAI 0.3 --R 0.5
    python eos_sensor.py --serve --port 5050

Author : M. Iizumi & T. Iizumi (Miosync, Inc.)
License: MIT
Version: 1.0.0 (2026-02-18)
"""

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============================================================================
# PHYSICAL CONSTANTS (from paper)
# ============================================================================

GAMMA_0 = math.cosh(1.0) ** 2          # 2.38109... canonical gate shape
SCALING_M = 0.814                       # α–β power-law exponent
SCALING_C = -1.355                      # α–β power-law intercept
DESIGN_K = 50.4                         # NRMSE = k · σ_SAI
S_BOUNDARY_R1R2 = 10.0                  # R1/R2 regime boundary
S_BOUNDARY_R2R3 = 3.0                   # R2/R3 regime boundary
FZ_LOW = 0.82                           # Forbidden zone lower bound
FZ_HIGH = 1.61                          # Forbidden zone upper bound


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def beta_pred(alpha: float) -> float:
    """Predict β from α via the population scaling law.

    log₁₀β = 0.814·log₁₀α − 1.355

    Parameters
    ----------
    alpha : float
        Light-harvesting efficiency (initial slope of PI curve).

    Returns
    -------
    float
        Predicted photoinhibition susceptibility β.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return 10 ** (SCALING_M * math.log10(alpha) + SCALING_C)


def classify_regime(S: float) -> dict:
    """Classify photosynthetic regime from gate variable S.

    Parameters
    ----------
    S : float
        Gate variable = α/β (irradiance-scale separation).

    Returns
    -------
    dict with keys:
        regime   : str   ("R1", "R2", "R3")
        label    : str   (human-readable)
        eos_tier : str   ("EOS2" or "EOS3")
        in_fz    : bool  (inside forbidden zone?)
    """
    in_fz = FZ_LOW < S < FZ_HIGH

    if S > S_BOUNDARY_R1R2:
        return {
            "regime": "R1",
            "label": "Factorized — PCC/SCC well separated",
            "eos_tier": "EOS2",
            "S": round(S, 2),
            "in_forbidden_zone": in_fz,
        }
    elif S > S_BOUNDARY_R2R3:
        return {
            "regime": "R2",
            "label": "Transition — SCC affects plateau",
            "eos_tier": "EOS3",
            "S": round(S, 2),
            "in_forbidden_zone": in_fz,
        }
    else:
        return {
            "regime": "R3",
            "label": "Coupled — outside EOS validity",
            "eos_tier": "NONE",
            "S": round(S, 2),
            "in_forbidden_zone": in_fz,
        }


def _safe_tanh_power(x: float, gamma: float) -> float:
    """Compute tanh(x^γ) with overflow protection."""
    if x <= 0:
        return 0.0
    try:
        val = x ** gamma
    except OverflowError:
        return 1.0
    if val > 20:
        return 1.0
    return math.tanh(val)


def pi_curve(
    I: float,
    Pmax: float,
    alpha: float,
    beta: float,
    gamma: float = GAMMA_0,
    R: float = 0.0,
) -> dict:
    """Evaluate PI model at a single irradiance.

    P_gross(I) = Pmax · tanh(αI/Pmax) · tanh((Pmax/(β·I))^γ)
    P_net(I)   = P_gross(I) − R

    Parameters
    ----------
    I     : Irradiance (µmol photons m⁻² s⁻¹)
    Pmax  : Maximum photosynthetic rate
    alpha : Light-harvesting efficiency
    beta  : Photoinhibition susceptibility
    gamma : SCC gate shape (default: cosh²(1))
    R     : Dark offset (default: 0)

    Returns
    -------
    dict with P_gross, P_net, PCC, SCC
    """
    if I <= 0:
        return {"I": I, "P_gross": 0.0, "P_net": -R, "PCC": 0.0, "SCC": 1.0}

    PCC = math.tanh(alpha * I / Pmax)
    SCC = _safe_tanh_power(Pmax / (beta * I), gamma)
    P_gross = Pmax * PCC * SCC
    P_net = P_gross - R

    return {
        "I": I,
        "P_gross": round(P_gross, 6),
        "P_net": round(P_net, 6),
        "PCC": round(PCC, 6),
        "SCC": round(SCC, 6),
    }


# ============================================================================
# SENSOR RESULT
# ============================================================================

@dataclass
class EOSResult:
    """Complete EOS prediction result."""

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

    # Quality metrics
    expected_NRMSE_pct: Optional[float]   # if σ_SAI known
    sigma_SAI: Optional[float]

    # PI curve (list of dicts)
    curve: list = field(default_factory=list)

    # Diagnostic
    I_alpha: float = 0.0     # PCC saturation irradiance
    I_beta: float = 0.0      # SCC onset irradiance
    I_opt: float = 0.0       # Optimum irradiance (from curve peak)

    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


# ============================================================================
# MAIN SENSOR CLASS
# ============================================================================

class EOSSensor:
    """Photosynthesis–Irradiance Equation of State Soft Sensor.

    Transforms minimal sensor readings into complete PI curve predictions
    with regime classification, error estimates, and diagnostic parameters.

    Examples
    --------
    >>> sensor = EOSSensor()

    # EOS2: just α and Pmax (e.g., from a quick 3-point RLC)
    >>> r = sensor.predict(alpha=0.05, Pmax=8.0)
    >>> print(r.regime, r.eos_tier)
    R1 EOS2

    # EOS3: add SAI for transition-regime accuracy
    >>> r = sensor.predict(alpha=0.05, Pmax=8.0, SAI=0.3)

    # With dark offset
    >>> r = sensor.predict(alpha=0.05, Pmax=8.0, R=0.5)
    """

    def __init__(
        self,
        I_range: tuple = (1, 2500),
        n_points: int = 200,
        gamma: float = GAMMA_0,
    ):
        self.I_min, self.I_max = I_range
        self.n_points = n_points
        self.gamma = gamma

    def predict(
        self,
        alpha: float,
        Pmax: float,
        SAI: float = None,
        R: float = 0.0,
        sigma_SAI: float = None,
        I_values: list = None,
    ) -> EOSResult:
        """Generate PI curve prediction from sensor inputs.

        Parameters
        ----------
        alpha     : Light-harvesting efficiency (from RLC initial slope)
        Pmax      : Maximum photosynthetic rate (from RLC plateau)
        SAI       : Stress Adaptation Index (optional; None → EOS2)
        R         : Dark offset (optional; default 0)
        sigma_SAI : Measurement uncertainty in SAI (for error estimate)
        I_values  : Custom irradiance array (optional)

        Returns
        -------
        EOSResult with full curve, regime info, diagnostics
        """
        # --- Validate ---
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if Pmax <= 0:
            raise ValueError(f"Pmax must be > 0, got {Pmax}")

        # --- β prediction ---
        bp = beta_pred(alpha)

        if SAI is not None:
            beta_eff = bp * (10 ** SAI)
            eos_mode = "EOS3"
        else:
            beta_eff = bp
            eos_mode = "EOS2"

        # --- Gate variable & regime ---
        S = alpha / beta_eff
        regime_info = classify_regime(S)

        # --- Characteristic irradiances ---
        I_alpha = Pmax / alpha            # PCC saturation
        I_beta = Pmax / beta_eff          # SCC onset

        # --- Generate curve ---
        if I_values is None:
            step = (self.I_max - self.I_min) / (self.n_points - 1)
            I_values = [self.I_min + i * step for i in range(self.n_points)]

        curve = [
            pi_curve(I, Pmax, alpha, beta_eff, self.gamma, R)
            for I in I_values
        ]

        # --- I_opt from actual curve peak (exact, γ₀-aware) ---
        peak = max(curve, key=lambda p: p["P_gross"])
        I_opt = peak["I"]

        # --- Error estimate ---
        if sigma_SAI is not None:
            expected_NRMSE = DESIGN_K * sigma_SAI
        elif regime_info["regime"] == "R1":
            # In R1, EOS2 is sufficient: typical NRMSE ≈ 2-5%
            expected_NRMSE = None
            sigma_SAI = None
        else:
            expected_NRMSE = None

        return EOSResult(
            alpha=alpha,
            Pmax=Pmax,
            SAI=SAI,
            R=R,
            beta_predicted=round(bp, 6),
            beta_effective=round(beta_eff, 6),
            S=round(S, 2),
            regime=regime_info["regime"],
            regime_label=regime_info["label"],
            eos_tier=eos_mode,
            in_forbidden_zone=regime_info["in_forbidden_zone"],
            expected_NRMSE_pct=round(expected_NRMSE, 2) if expected_NRMSE else None,
            sigma_SAI=sigma_SAI,
            curve=curve,
            I_alpha=round(I_alpha, 1),
            I_beta=round(I_beta, 1),
            I_opt=round(I_opt, 1),
        )

    def diagnose(self, alpha: float, Pmax: float, beta_obs: float) -> dict:
        """Compute SAI and regime from observed parameters.

        Use when you have a full PI curve fit and want to classify it.

        Parameters
        ----------
        alpha    : Observed α
        Pmax     : Observed Pmax
        beta_obs : Observed β (from full PI curve fit)

        Returns
        -------
        dict with SAI, S, regime, recommendations
        """
        bp = beta_pred(alpha)
        SAI = math.log10(beta_obs) - math.log10(bp)
        S = alpha / beta_obs
        regime_info = classify_regime(S)

        # Recommendations
        recs = []
        if regime_info["regime"] == "R1":
            recs.append("EOS2 sufficient. SAI available but not required for prediction.")
        elif regime_info["regime"] == "R2":
            recs.append("EOS3 recommended. Include SAI for accurate prediction.")
            if SAI > 0.5:
                recs.append("⚠ HIGH SAI: elevated photoinhibition risk. Consider reducing irradiance.")
            elif SAI < -0.5:
                recs.append("Low SAI: strong photoprotection/acclimation. Culture well-adapted.")
        else:
            recs.append("⚠ R3 regime: EOS not valid. Full curve fitting required.")

        if regime_info["in_forbidden_zone"]:
            recs.append("⚠ In forbidden zone: structural transition detected. Monitor closely.")

        return {
            "SAI": round(SAI, 4),
            "S": round(S, 2),
            "beta_predicted": round(bp, 6),
            "beta_observed": beta_obs,
            **regime_info,
            "recommendations": recs,
        }

    def design_spec(self, target_NRMSE_pct: float) -> dict:
        """Compute instrument requirements for a target accuracy.

        Parameters
        ----------
        target_NRMSE_pct : Desired NRMSE in percent (e.g., 5.0)

        Returns
        -------
        dict with required σ_SAI and practical implications
        """
        sigma_req = target_NRMSE_pct / DESIGN_K
        return {
            "target_NRMSE_pct": target_NRMSE_pct,
            "required_sigma_SAI": round(sigma_req, 4),
            "design_law": f"NRMSE = {DESIGN_K} × σ_SAI",
            "note": (
                f"To achieve NRMSE < {target_NRMSE_pct}%, "
                f"SAI measurement uncertainty must be < {sigma_req:.3f}. "
                f"In silico RLC simulations indicate σ(SAI) ≈ 0.05 is achievable."
            ),
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EOS Soft Sensor — PI curve prediction from minimal inputs",
        epilog="Example: python eos_sensor.py --alpha 0.05 --Pmax 8.0 --SAI 0.2",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Light-harvesting efficiency")
    parser.add_argument("--Pmax", type=float, default=None, help="Maximum photosynthetic rate")
    parser.add_argument("--SAI", type=float, default=None, help="Stress Adaptation Index (optional)")
    parser.add_argument("--R", type=float, default=0.0, help="Dark offset (default: 0)")
    parser.add_argument("--sigma-SAI", type=float, default=None, help="SAI uncertainty for error estimate")
    parser.add_argument("--beta-obs", type=float, default=None, help="Observed β for diagnosis mode")
    parser.add_argument("--target-NRMSE", type=float, default=None, help="Target NRMSE for design spec")
    parser.add_argument("--json", action="store_true", help="Output full JSON (including curve)")
    parser.add_argument("--compact", action="store_true", help="Suppress curve in output")
    parser.add_argument("--n-points", type=int, default=50, help="Number of irradiance points")
    parser.add_argument("--serve", action="store_true", help="Start Flask API server")
    parser.add_argument("--port", type=int, default=5050, help="API server port (default: 5050)")

    args = parser.parse_args()

    # --- Server mode ---
    if args.serve:
        serve(port=args.port)
        return

    # --- All other modes require alpha & Pmax ---
    if args.alpha is None or args.Pmax is None:
        parser.error("--alpha and --Pmax are required (unless using --serve)")

    sensor = EOSSensor(n_points=args.n_points)

    # --- Design spec mode ---
    if args.target_NRMSE is not None:
        spec = sensor.design_spec(args.target_NRMSE)
        print(json.dumps(spec, indent=2))
        return

    # --- Diagnosis mode ---
    if args.beta_obs is not None:
        diag = sensor.diagnose(args.alpha, args.Pmax, args.beta_obs)
        print(json.dumps(diag, indent=2))
        return

    # --- Prediction mode ---
    result = sensor.predict(
        alpha=args.alpha,
        Pmax=args.Pmax,
        SAI=args.SAI,
        R=args.R,
        sigma_SAI=args.sigma_SAI,
    )

    if args.json:
        if args.compact:
            d = asdict(result)
            d.pop("curve")
            print(json.dumps(d, indent=2))
        else:
            print(result.to_json())
    else:
        # Human-readable summary
        print("=" * 60)
        print("  EOS SOFT SENSOR — PI Curve Prediction")
        print("=" * 60)
        print(f"  Mode      : {result.eos_tier}")
        print(f"  Regime    : {result.regime} ({result.regime_label})")
        print(f"  S (gate)  : {result.S}")
        if result.in_forbidden_zone:
            print(f"  ⚠ WARNING : Inside forbidden zone!")
        print("-" * 60)
        print(f"  α (input) : {result.alpha}")
        print(f"  Pmax (in) : {result.Pmax}")
        if result.SAI is not None:
            print(f"  SAI (in)  : {result.SAI}")
        print(f"  R (offset): {result.R}")
        print("-" * 60)
        print(f"  β_pred    : {result.beta_predicted}")
        print(f"  β_eff     : {result.beta_effective}")
        print(f"  I_α       : {result.I_alpha} µmol m⁻² s⁻¹")
        print(f"  I_β       : {result.I_beta} µmol m⁻² s⁻¹")
        print(f"  I_opt     : {result.I_opt} µmol m⁻² s⁻¹")
        if result.expected_NRMSE_pct:
            print(f"  Expected NRMSE : {result.expected_NRMSE_pct}%")
        print("=" * 60)

        # Print curve table (sampled)
        print(f"\n{'I':>8}  {'P_gross':>10}  {'P_net':>10}  {'PCC':>8}  {'SCC':>8}")
        print("-" * 52)
        step = max(1, len(result.curve) // 20)
        for i in range(0, len(result.curve), step):
            p = result.curve[i]
            print(f"{p['I']:8.1f}  {p['P_gross']:10.4f}  {p['P_net']:10.4f}  {p['PCC']:8.4f}  {p['SCC']:8.4f}")


if __name__ == "__main__":
    main()


# ============================================================================
# FLASK API EXAMPLE (optional — run with `python eos_sensor.py serve`)
# ============================================================================
# To use: pip install flask, then:
#   python -c "from eos_sensor import serve; serve()"
#
# POST /predict  {"alpha": 0.05, "Pmax": 8.0, "SAI": null, "R": 0}
# POST /diagnose {"alpha": 0.05, "Pmax": 8.0, "beta_obs": 0.01}
# POST /design   {"target_NRMSE_pct": 5.0}
# ============================================================================

def serve(host="0.0.0.0", port=5050):
    """Start Flask API server."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return

    app = Flask(__name__)
    sensor = EOSSensor(n_points=100)

    @app.route("/predict", methods=["POST"])
    def api_predict():
        d = request.get_json()
        try:
            result = sensor.predict(
                alpha=d["alpha"],
                Pmax=d["Pmax"],
                SAI=d.get("SAI"),
                R=d.get("R", 0),
                sigma_SAI=d.get("sigma_SAI"),
            )
            return jsonify(asdict(result))
        except (ValueError, KeyError) as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/diagnose", methods=["POST"])
    def api_diagnose():
        d = request.get_json()
        try:
            result = sensor.diagnose(d["alpha"], d["Pmax"], d["beta_obs"])
            return jsonify(result)
        except (ValueError, KeyError) as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/design", methods=["POST"])
    def api_design():
        d = request.get_json()
        try:
            result = sensor.design_spec(d["target_NRMSE_pct"])
            return jsonify(result)
        except (ValueError, KeyError) as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "version": "1.0.0",
            "constants": {
                "gamma_0": GAMMA_0,
                "scaling_m": SCALING_M,
                "scaling_c": SCALING_C,
                "design_k": DESIGN_K,
            }
        })

    print(f"EOS Soft Sensor API running on {host}:{port}")
    print("Endpoints: POST /predict, POST /diagnose, POST /design, GET /health")
    app.run(host=host, port=port)
