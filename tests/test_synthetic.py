import numpy as np
import pytest

from src import data, synthetic


@pytest.mark.parametrize("effect", ["constant", "heterogeneous"])
def test_simulate_has_criteo_schema_and_design(effect):
    df = synthetic.simulate(n=50_000, effect=effect, seed=1)
    data.validate(df)  # binary columns, no exposure in control, no missing values
    assert df["treatment"].mean() == pytest.approx(0.85, abs=0.01)
    assert set(synthetic.TRUTH_COLS) <= set(df.columns)


def test_simulate_is_reproducible():
    a = synthetic.simulate(n=1_000, seed=3)
    b = synthetic.simulate(n=1_000, seed=3)
    assert a.equals(b)


def test_constant_effect_is_constant():
    df = synthetic.simulate(n=10_000, effect="constant", tau=0.2, seed=2)
    assert np.allclose(df["tau_exposed"], 0.2)
    assert synthetic.true_effects(df)["itt"] == pytest.approx(0.3 * 0.2)


def test_heterogeneous_effect_varies_but_keeps_mean():
    df = synthetic.simulate(n=200_000, effect="heterogeneous", tau=0.2, seed=2)
    assert df["tau_exposed"].std() > 0.05
    assert df["tau_exposed"].mean() == pytest.approx(0.2, rel=0.02)
    # responders sit at high f0
    hi = df.loc[df["f0"] > 1, "tau_exposed"].mean()
    lo = df.loc[df["f0"] < -1, "tau_exposed"].mean()
    assert hi > 2 * lo


@pytest.mark.parametrize("effect", ["constant", "heterogeneous"])
def test_difference_in_means_recovers_true_itt(effect):
    """The fixture is only useful if randomization works: diff in means hits the truth."""
    df = synthetic.simulate(n=400_000, effect=effect, seed=4)
    t = df["treatment"] == 1
    diff = df.loc[t, "visit"].mean() - df.loc[~t, "visit"].mean()
    se = np.sqrt(df.loc[t, "visit"].var() / t.sum() + df.loc[~t, "visit"].var() / (~t).sum())
    assert abs(diff - synthetic.true_effects(df)["itt"]) < 4 * se


def test_wald_ratio_recovers_effect_on_exposed():
    df = synthetic.simulate(n=400_000, effect="constant", tau=0.2, seed=5)
    t = df["treatment"] == 1
    itt = df.loc[t, "visit"].mean() - df.loc[~t, "visit"].mean()
    wald = itt / df.loc[t, "exposure"].mean()
    assert wald == pytest.approx(0.2, abs=0.02)


def test_simulate_rejects_unknown_effect():
    with pytest.raises(ValueError, match="effect must be"):
        synthetic.simulate(n=10, effect="linear")
