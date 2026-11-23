"""Experiment analysis on synthetic data where the true effect is known."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src import experiment, synthetic

# ----------------------------------------------------------------------------- SRM


def test_srm_passes_at_design_split_and_fails_against_half():
    ok = experiment.srm_test(850_000, 150_000, p_treated=0.85)
    assert ok["chi2"] == pytest.approx(0.0) and not ok["srm_flag"]
    # the same counts against the wrong null of 50/50 must fail
    assert experiment.srm_test(850_000, 150_000, p_treated=0.5)["srm_flag"]


def test_srm_detects_small_shift_at_scale():
    # 0.849 vs 0.85 on 14M users is a real mismatch
    n = 14_000_000
    nt = int(0.849 * n)
    r = experiment.srm_test(nt, n - nt)
    assert r["srm_flag"] and r["p_value"] < 1e-10


def test_srm_matches_hand_chi_square():
    nt, nc = 8_600, 1_400
    hand = (nt - 8_500) ** 2 / 8_500 + (nc - 1_500) ** 2 / 1_500
    assert experiment.srm_test(nt, nc)["chi2"] == pytest.approx(hand)


def test_srm_false_alarm_rate_under_correct_design():
    rng = np.random.default_rng(0)
    n = 100_000
    flags = [
        experiment.srm_test(nt, n - nt)["p_value"] < 0.05 for nt in rng.binomial(n, 0.85, 2000)
    ]
    assert np.mean(flags) == pytest.approx(0.05, abs=0.015)


# ------------------------------------------------------------------------- balance


@pytest.fixture(scope="module")
def sim():
    return synthetic.simulate(n=100_000, effect="heterogeneous", seed=11)


def test_balance_randomized_data_is_balanced(sim):
    b = experiment.balance_table(sim)
    assert not b["smd_flag"].any() and not b["vr_flag"].any()
    assert b["variance_ratio"].between(0.9, 1.1).all()


def test_balance_matches_hand_computation(sim):
    b = experiment.balance_table(sim).set_index("feature")
    t = sim["treatment"] == 1
    x_t, x_c = sim.loc[t, "f4"], sim.loc[~t, "f4"]
    smd = (x_t.mean() - x_c.mean()) / np.sqrt((x_t.var() + x_c.var()) / 2)
    assert b.loc["f4", "smd"] == pytest.approx(smd)
    assert b.loc["f4", "variance_ratio"] == pytest.approx(x_t.var() / x_c.var())


def test_balance_flags_mean_shift(sim):
    df = sim.copy()
    df.loc[df["treatment"] == 1, "f3"] += 0.3
    b = experiment.balance_table(df).set_index("feature")
    assert b.loc["f3", "smd"] == pytest.approx(0.3, abs=0.03)
    assert b.loc["f3", "smd_flag"] and b["smd_flag"].sum() == 1


def test_balance_variance_ratio_catches_what_smd_misses(sim):
    df = sim.copy()
    t = df["treatment"] == 1
    df.loc[t, "f5"] = df.loc[t, "f5"] * 1.6  # same mean (0), 2.56x the variance
    b = experiment.balance_table(df).set_index("feature")
    assert abs(b.loc["f5", "smd"]) < 0.1 and not b.loc["f5", "smd_flag"]
    assert b.loc["f5", "variance_ratio"] == pytest.approx(2.56, rel=0.05)
    assert b.loc["f5", "vr_flag"]


# ----------------------------------------------------------------------------- ITT


@pytest.mark.parametrize("effect", ["constant", "heterogeneous"])
def test_itt_recovers_true_effect(effect):
    df = synthetic.simulate(n=400_000, effect=effect, seed=12)
    r = experiment.itt(df, "visit")
    truth = synthetic.true_effects(df)["itt"]
    assert r["abs_ci_low"] < truth < r["abs_ci_high"]
    assert abs(r["abs_lift"] - truth) < 4 * r["abs_se"]


def test_itt_matches_welch_t_test(sim):
    r = experiment.itt(sim, "visit")
    t = sim["treatment"] == 1
    w = stats.ttest_ind(sim.loc[t, "visit"], sim.loc[~t, "visit"], equal_var=False)
    assert r["abs_lift"] / r["abs_se"] == pytest.approx(w.statistic, rel=1e-6)
    assert r["p_value"] == pytest.approx(w.pvalue, rel=1e-3)


def test_itt_ci_coverage_is_nominal():
    """Across repeated experiments the 95% CIs must cover the truth about 95% of the time."""
    hits_abs, hits_rel = [], []
    for seed in range(300):
        df = synthetic.simulate(n=20_000, effect="heterogeneous", seed=1000 + seed)
        r = experiment.itt(df, "visit")
        truth = synthetic.true_effects(df)["itt"]
        truth_rel = truth / df["p_visit0"].mean()
        hits_abs.append(r["abs_ci_low"] < truth < r["abs_ci_high"])
        hits_rel.append(r["rel_ci_low"] < truth_rel < r["rel_ci_high"])
    # 300 draws: binomial SD of coverage is about 0.0126
    assert np.mean(hits_abs) == pytest.approx(0.95, abs=0.035)
    assert np.mean(hits_rel) == pytest.approx(0.95, abs=0.035)


def test_itt_false_positive_rate_under_null():
    pvals = [
        experiment.itt(synthetic.simulate(n=10_000, tau=0.0, seed=2000 + s), "visit")["p_value"]
        for s in range(300)
    ]
    assert np.mean(np.array(pvals) < 0.05) == pytest.approx(0.05, abs=0.035)


# ----------------------------------------------------------------------------- MDE


def test_mde_matches_formula():
    n_t, n_c, var = 850_000, 150_000, 0.04 * 0.96
    se = np.sqrt(var * (1 / n_t + 1 / n_c))
    assert experiment.mde(n_t, n_c, var) == pytest.approx(
        (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se
    )
    assert experiment.mde(n_t, n_c, var) == pytest.approx(2.80 * se, rel=0.002)


def test_mde_shrinks_with_n_and_grows_with_power():
    base = experiment.mde(8_500, 1_500, 0.04)
    assert experiment.mde(34_000, 6_000, 0.04) == pytest.approx(base / 2)
    assert experiment.mde(8_500, 1_500, 0.04, power=0.9) > base


def test_mde_delivers_80_percent_power():
    """Simulate experiments whose true lift equals the MDE: about 80% should reject."""
    rng = np.random.default_rng(3)
    p0, n_t, n_c = 0.04, 85_000, 15_000
    m = experiment.mde_binary(p0, n_t, n_c)
    assert m["mde_rel"] == pytest.approx(m["mde_abs"] / p0)
    reps = 4_000
    y_t = rng.binomial(n_t, p0 + m["mde_abs"], reps) / n_t
    y_c = rng.binomial(n_c, p0, reps) / n_c
    se = np.sqrt(y_t * (1 - y_t) / n_t + y_c * (1 - y_c) / n_c)
    power = np.mean(np.abs((y_t - y_c) / se) > stats.norm.ppf(0.975))
    assert power == pytest.approx(0.80, abs=0.03)


# ---------------------------------------------------------------------------- Holm


@pytest.mark.parametrize(
    "p", [[0.01, 0.04, 0.03], [0.2, 0.001], [0.5, 0.5, 0.01, 0.02], [0.04, 0.04]]
)
def test_holm_matches_statsmodels(p):
    ours = experiment.holm(p)
    reject, adj, _, _ = multipletests(p, alpha=0.05, method="holm")
    np.testing.assert_allclose(ours["p_adjusted"], adj)
    np.testing.assert_array_equal(ours["reject"], reject)


def test_holm_by_hand():
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 -> 0.03, 0.06, 0.06 (running max)
    r = experiment.holm([0.04, 0.01, 0.03])
    np.testing.assert_allclose(r["p_adjusted"], [0.06, 0.03, 0.06])
    assert r["reject"].tolist() == [False, True, False]


# ------------------------------------------------------------------------ pipeline


def test_analyze_end_to_end(sim):
    res = experiment.analyze(sim)
    t = res["itt"]
    assert t["outcome"].tolist() == ["visit", "conversion"]  # visit leads
    assert (t["p_holm"] >= t["p_value"]).all()
    assert (t["mde_abs"] > 0).all()
    assert not res["srm"]["srm_flag"]
    assert len(res["balance"]) == 12
    pm = experiment.pm_table(t)
    assert list(pm["Outcome"]) == ["Visit", "Conversion"]
    assert isinstance(pm, pd.DataFrame)
