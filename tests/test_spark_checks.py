"""Spark checks on synthetic data: every number must match a pandas hand computation."""

import json

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from src import data, spark_checks, synthetic  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = spark_checks.get_spark("test-checks", driver_memory="1g", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture(scope="module")
def sim():
    return synthetic.simulate(n=20_000, effect="heterogeneous", seed=7)


@pytest.fixture(scope="module")
def sdf(spark, sim):
    return spark.createDataFrame(sim[data.COLUMNS])


def test_counts_and_rates_match_pandas(sdf, sim):
    s = spark_checks.counts_and_rates(sdf)
    p = data.summarize(sim)
    assert all(spark_checks.compare_with_pandas(s, p).values())
    t = sim["treatment"] == 1
    assert s["visit_rate_treated"] == pytest.approx(sim.loc[t, "visit"].mean())
    assert s["visit_rate_control"] == pytest.approx(sim.loc[~t, "visit"].mean())
    assert s["conversion_rate_control"] == pytest.approx(sim.loc[~t, "conversion"].mean())
    assert s["n_exposed_control"] == 0
    json.dumps(s)


def test_counts_and_rates_catch_exposed_control(spark, sim):
    bad = sim[data.COLUMNS].copy()
    bad.loc[bad.index[bad["treatment"] == 0][:3], "exposure"] = 1
    assert spark_checks.counts_and_rates(spark.createDataFrame(bad))["n_exposed_control"] == 3


def test_feature_means_by_arm_match_pandas(sdf, sim):
    m = spark_checks.feature_means_by_arm(sdf).set_index("feature")
    g = sim.groupby("treatment")[data.FEATURES]
    mean, sd = g.mean(), g.std()
    np.testing.assert_allclose(m["mean_treated"], mean.loc[1], rtol=1e-9)
    np.testing.assert_allclose(m["mean_control"], mean.loc[0], rtol=1e-9)
    np.testing.assert_allclose(m["sd_control"], sd.loc[0], rtol=1e-9)
    smd = (mean.loc[1] - mean.loc[0]) / np.sqrt((sd.loc[1] ** 2 + sd.loc[0] ** 2) / 2)
    np.testing.assert_allclose(m["smd"], smd, rtol=1e-9)
    # randomized features: balance holds
    assert m["smd"].abs().max() < 0.1


def test_feature_means_flag_imbalance(spark, sim):
    shifted = sim[data.COLUMNS].copy()
    shifted.loc[shifted["treatment"] == 1, "f3"] += 0.5
    m = spark_checks.feature_means_by_arm(spark.createDataFrame(shifted)).set_index("feature")
    assert m.loc["f3", "smd"] == pytest.approx(0.5, abs=0.05)


def test_compare_with_pandas_flags_mismatch():
    res = spark_checks.compare_with_pandas(
        {"n_rows": 10, "visit_rate": 0.05}, {"n_rows": 11, "visit_rate": 0.05}
    )
    assert res == {"n_rows": False, "visit_rate": True}


def test_run_end_to_end(tmp_path, spark, sim):
    full = tmp_path / "full.parquet"
    sim[data.COLUMNS].to_parquet(full)
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(json.dumps({"full": data.summarize(sim)}))
    cfg = data.load_config()
    cfg["data"].update(full_parquet=str(full), summary_json=str(summary_json))
    cfg["expected"] = dict(cfg["expected"], n_rows=len(sim))
    res = spark_checks.run(cfg, spark=spark, tables_dir=tmp_path / "tables")
    assert res["published_checks"]["n_rows"]
    assert all(res["pandas_checks"].values())
    assert isinstance(res["feature_means"], pd.DataFrame)
    assert json.loads((tmp_path / "spark_checks.json").read_text())["srm"]["p_treated_design"]
