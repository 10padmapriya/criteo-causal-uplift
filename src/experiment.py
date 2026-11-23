"""Analyze the data as a randomized experiment.

Day 3: SRM, covariate balance (SMD and variance ratio), ITT with 95% CIs on the
absolute and relative scale, MDE at 80% power, Holm correction across outcomes.
Day 4 adds CUPED, the effect on the exposed and the peeking simulation.

Run on the 10% sample with ``python -m src.experiment`` and on the full data
with ``python -m src.experiment --full`` (``make experiment``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import FEATURES, ROOT, load, load_config

OUTCOMES = ["visit", "conversion"]  # visit leads: primary outcome first


# --------------------------------------------------------------------------- SRM
def srm_test(n_treated: int, n_control: int, p_treated: float = 0.85) -> dict:
    """Chi square goodness of fit of the observed split against the design split.

    The design is 85/15, so the null is ``p_treated = 0.85``, never 0.50. A small
    p value (convention: below 0.001) means assignment or logging is broken and no
    downstream effect estimate should be trusted.
    """
    n = n_treated + n_control
    expected = [n * p_treated, n * (1 - p_treated)]
    chi2, p = stats.chisquare([n_treated, n_control], f_exp=expected)
    return {
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "p_treated_design": p_treated,
        "p_treated_observed": n_treated / n,
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_flag": bool(p < 0.001),
    }


# ----------------------------------------------------------------------- balance
def balance_table(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    treatment: str = "treatment",
    smd_threshold: float = 0.1,
    vr_bounds: tuple[float, float] = (0.5, 2.0),
) -> pd.DataFrame:
    """Per feature balance between arms.

    * ``smd``: ``(mean_T - mean_C) / sqrt((var_T + var_C) / 2)``, flagged when
      ``|smd| > smd_threshold`` (0.1 is the usual rule).
    * ``variance_ratio``: ``var_T / var_C``, flagged outside ``vr_bounds``
      (Rubin's 0.5 to 2). SMD only compares centres; two arms can share a mean
      and still have different spreads, which this catches.

    No p values on purpose: with millions of rows any difference is significant,
    so significance says nothing about whether the imbalance matters.
    """
    t = df[treatment] == 1
    g_t, g_c = df.loc[t, features], df.loc[~t, features]
    out = pd.DataFrame(
        {
            "mean_treated": g_t.mean(),
            "mean_control": g_c.mean(),
            "var_treated": g_t.var(),
            "var_control": g_c.var(),
        }
    )
    out["smd"] = (out["mean_treated"] - out["mean_control"]) / np.sqrt(
        (out["var_treated"] + out["var_control"]) / 2
    )
    out["variance_ratio"] = out["var_treated"] / out["var_control"]
    out["smd_flag"] = out["smd"].abs() > smd_threshold
    lo, hi = vr_bounds
    out["vr_flag"] = (out["variance_ratio"] < lo) | (out["variance_ratio"] > hi)
    return out.rename_axis("feature").reset_index()


# ----------------------------------------------------------------------------- ITT
def itt(df: pd.DataFrame, outcome: str, treatment: str = "treatment", alpha: float = 0.05) -> dict:
    """Intent to treat effect: difference in mean outcome by assigned arm.

    Uses the unpooled (Neyman) standard error
    ``sqrt(var_T / n_T + var_C / n_C)``, which stays valid when the arms have
    different variances and very different sizes (85/15). The CI is a normal
    approximation; with thousands of events per arm the CLT holds comfortably.

    The relative lift ``(mean_T - mean_C) / mean_C`` gets a delta method SE:
    ``var(r) ~= var_T/n_T / mean_C^2 + mean_T^2 * var_C/n_C / mean_C^4``.
    """
    t = df[treatment] == 1
    y_t, y_c = df.loc[t, outcome], df.loc[~t, outcome]
    n_t, n_c = len(y_t), len(y_c)
    m_t, m_c = float(y_t.mean()), float(y_c.mean())
    v_t, v_c = float(y_t.var()), float(y_c.var())
    se_t2, se_c2 = v_t / n_t, v_c / n_c

    diff = m_t - m_c
    se = np.sqrt(se_t2 + se_c2)
    z = stats.norm.ppf(1 - alpha / 2)
    rel = diff / m_c
    se_rel = np.sqrt(se_t2 / m_c**2 + m_t**2 * se_c2 / m_c**4)
    p = 2 * stats.norm.sf(abs(diff / se))
    return {
        "outcome": outcome,
        "n_treated": n_t,
        "n_control": n_c,
        "mean_treated": m_t,
        "mean_control": m_c,
        "abs_lift": diff,
        "abs_se": float(se),
        "abs_ci_low": float(diff - z * se),
        "abs_ci_high": float(diff + z * se),
        "rel_lift": float(rel),
        "rel_se": float(se_rel),
        "rel_ci_low": float(rel - z * se_rel),
        "rel_ci_high": float(rel + z * se_rel),
        "z": float(diff / se),
        "p_value": float(p),
    }


# ----------------------------------------------------------------------------- MDE
def mde(
    n_treated: int, n_control: int, variance: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Minimum detectable absolute effect for a two sided test.

    ``(z_{1-alpha/2} + z_{power}) * sqrt(variance * (1/n_T + 1/n_C))``, which is
    ``(1.96 + 0.84) * SE`` at the defaults. An effect this large is detected 80%
    of the time; smaller true effects usually are not.
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(variance * (1 / n_treated + 1 / n_control)))


def mde_binary(
    baseline: float, n_treated: int, n_control: int, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """MDE for a 0/1 outcome, using the control rate's Bernoulli variance p(1 - p)."""
    abs_mde = mde(n_treated, n_control, baseline * (1 - baseline), alpha, power)
    return {"baseline": baseline, "mde_abs": abs_mde, "mde_rel": abs_mde / baseline}


# ---------------------------------------------------------------------------- Holm
def holm(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Holm step down adjustment for a family of tests.

    Sort p values ascending; the k-th smallest (k from 0) is multiplied by
    ``m - k``; adjusted values are made monotone with a running max and capped
    at 1. Controls the family wise error rate like Bonferroni but is never less
    powerful. Returns adjusted p values and reject decisions in the input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {"p_adjusted": adj, "reject": adj <= alpha}


# ------------------------------------------------------------------------ pipeline
def analyze(
    df: pd.DataFrame,
    outcomes: list[str] = OUTCOMES,
    p_treated: float = 0.85,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """SRM, balance and an ITT results table (one row per outcome, visit first)."""
    n_t = int((df["treatment"] == 1).sum())
    n_c = len(df) - n_t
    rows = []
    for y in outcomes:
        r = itt(df, y, alpha=alpha)
        m = mde_binary(r["mean_control"], n_t, n_c, alpha, power)
        r.update(mde_abs=m["mde_abs"], mde_rel=m["mde_rel"])
        rows.append(r)
    table = pd.DataFrame(rows)
    h = holm(table["p_value"], alpha)
    table["p_holm"] = h["p_adjusted"]
    table["significant_holm"] = h["reject"]
    return {
        "srm": srm_test(n_t, n_c, p_treated),
        "balance": balance_table(df),
        "itt": table,
    }


def pm_table(itt_table: pd.DataFrame) -> pd.DataFrame:
    """The ITT table in percentages for a non technical reader."""

    def pct(x: float, d: int = 2) -> str:
        return f"{100 * x:.{d}f}%"

    return pd.DataFrame(
        {
            "Outcome": itt_table["outcome"].str.capitalize(),
            "Control rate": itt_table["mean_control"].map(lambda x: pct(x, 3)),
            "Treated rate": itt_table["mean_treated"].map(lambda x: pct(x, 3)),
            "Lift (pts)": itt_table.apply(
                lambda r: (
                    f"{100 * r.abs_lift:+.3f} [{100 * r.abs_ci_low:.3f}, {100 * r.abs_ci_high:.3f}]"
                ),
                axis=1,
            ),
            "Relative lift": itt_table.apply(
                lambda r: f"{pct(r.rel_lift, 1)} [{pct(r.rel_ci_low, 1)}, {pct(r.rel_ci_high, 1)}]",
                axis=1,
            ),
            "Holm p": itt_table["p_holm"].map(lambda x: f"{x:.2g}" if x > 0 else "<1e-300"),
            "MDE (rel)": itt_table["mde_rel"].map(lambda x: pct(x, 1)),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 3 experiment analysis")
    ap.add_argument("--full", action="store_true", help="use the full data, not the 10% sample")
    args = ap.parse_args()
    cfg = load_config()
    which = "full" if args.full else "sample"
    df = load(which, columns=FEATURES + ["treatment", "visit", "conversion"], cfg=cfg)
    exp = cfg["experiment"]
    res = analyze(df, p_treated=exp["treated_share_design"], alpha=exp["alpha"], power=exp["power"])

    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    res["itt"].to_csv(tables / f"itt_{which}.csv", index=False, float_format="%.6g")
    res["balance"].to_csv(tables / f"balance_{which}.csv", index=False, float_format="%.6g")
    Path(cfg["data"]["summary_json"]).with_name(f"experiment_{which}.json").write_text(
        json.dumps(
            {"srm": res["srm"], "itt": res["itt"].to_dict(orient="records")}, indent=2, default=str
        )
    )
    print(f"== {which}: {len(df):,} rows")
    print("SRM:", json.dumps(res["srm"]))
    print(res["balance"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(pm_table(res["itt"]).to_string(index=False))


if __name__ == "__main__":
    main()
