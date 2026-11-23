"""Synthetic Criteo-like data with a known treatment effect, for tests and simulations.

The generator mirrors the real design: 85/15 random assignment, one sided
noncompliance (only treated users can be exposed), and an effect on visit that
flows only through exposure. Because the truth is known, every estimator can be
checked against it before it touches real data.

Ground truth columns returned alongside the Criteo schema:

* ``tau_exposed``: effect of seeing an ad on the visit probability, per user.
* ``tau_itt``: effect of being assigned to treatment, ``exposure_rate * tau_exposed``.
* ``p_visit0``: visit probability without treatment (the control potential outcome).

Never pass these to an estimator; they exist only to score it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURES

TRUTH_COLS = ["tau_exposed", "tau_itt", "p_visit0"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(
    n: int = 50_000,
    effect: str = "constant",
    tau: float = 0.20,
    exposure_rate: float = 0.30,
    treated_share: float = 0.85,
    base_visit: float = 0.04,
    conversion_given_visit: float = 0.06,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw ``n`` users with Criteo's columns plus ground truth.

    Args:
        effect: ``"constant"`` gives every user ``tau_exposed == tau``.
            ``"heterogeneous"`` makes it depend on ``f0`` and ``f2`` while keeping
            its mean at ``tau``, so the average effect is the same in both modes.
        tau: mean effect of exposure on the visit probability.
        exposure_rate: P(exposure = 1 | treated). Independent of features, so the
            true ITT is exactly ``exposure_rate * mean(tau_exposed)``.
        base_visit: approximate control visit rate. Depends on ``f0`` and ``f1``
            so that covariate adjustment (CUPED) has something to remove.
    """
    if effect not in {"constant", "heterogeneous"}:
        raise ValueError(f"effect must be 'constant' or 'heterogeneous', got {effect!r}")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)

    logit0 = np.log(base_visit / (1 - base_visit)) + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    p_visit0 = _sigmoid(logit0)

    if effect == "constant":
        tau_exposed = np.full(n, tau)
    else:
        # Responders sit at high f0 and low f2. Rescale so the population mean is tau.
        raw = _sigmoid(1.5 * X[:, 0] - 1.0 * X[:, 2])
        tau_exposed = tau * raw / raw.mean()
    # Keep every probability inside [0, 1].
    tau_exposed = np.minimum(tau_exposed, 1.0 - p_visit0)

    treatment = rng.random(n) < treated_share
    exposure = treatment & (rng.random(n) < exposure_rate)
    p_visit = p_visit0 + exposure * tau_exposed
    visit = rng.random(n) < p_visit
    conversion = visit & (rng.random(n) < conversion_given_visit)

    df["treatment"] = treatment.astype("int8")
    df["conversion"] = conversion.astype("int8")
    df["visit"] = visit.astype("int8")
    df["exposure"] = exposure.astype("int8")
    df["tau_exposed"] = tau_exposed
    df["tau_itt"] = exposure_rate * tau_exposed
    df["p_visit0"] = p_visit0
    return df


def true_effects(df: pd.DataFrame) -> dict[str, float]:
    """Population truth for a simulated frame: ITT and effect on the exposed."""
    return {
        "itt": float(df["tau_itt"].mean()),
        "effect_on_exposed": float(df["tau_exposed"].mean()),
    }
