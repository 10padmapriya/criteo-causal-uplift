"""Full data checks in PySpark: counts, rates, SRM, feature means by arm (Day 2).

Run with ``make spark`` (``python -m src.spark_checks``). It reads the full
Parquet, checks it against Criteo's published values, cross checks every number
against the pandas summary written by ``make data``, and writes
``data/spark_checks.json`` plus ``reports/tables/feature_means_by_arm.csv``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from scipy import stats

from src.data import FEATURES, ROOT, check_expected, load_config


def get_spark(
    app_name: str = "criteo", driver_memory: str = "8g", shuffle_partitions: int = 64
) -> SparkSession:
    """Local SparkSession. Needs Java 17 on PATH or JAVA_HOME.

    Pins Spark's Python workers to this interpreter; otherwise they pick up
    whatever ``python3`` is on PATH and fail with PYTHON_VERSION_MISMATCH.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def counts_and_rates(sdf: DataFrame) -> dict:
    """Headline counts and rates in one pass, overall and by arm.

    Keys match ``src.data.summarize`` so the two can be compared directly.
    ``n_exposed_control`` must be 0 (one sided noncompliance).
    """
    t = F.col("treatment") == 1
    c = F.col("treatment") == 0
    row = sdf.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.sum(t.cast("long")).alias("n_treated"),
        F.avg("visit").alias("visit_rate"),
        F.avg("conversion").alias("conversion_rate"),
        F.avg(F.when(t, F.col("visit"))).alias("visit_rate_treated"),
        F.avg(F.when(c, F.col("visit"))).alias("visit_rate_control"),
        F.avg(F.when(t, F.col("conversion"))).alias("conversion_rate_treated"),
        F.avg(F.when(c, F.col("conversion"))).alias("conversion_rate_control"),
        F.avg(F.when(t, F.col("exposure"))).alias("exposure_rate_treated"),
        F.sum(F.when(c, F.col("exposure")).otherwise(0).cast("long")).alias("n_exposed_control"),
    ).first()
    out = row.asDict()
    for k in ("n_rows", "n_treated", "n_exposed_control"):
        out[k] = int(out[k])
    out["n_control"] = out["n_rows"] - out["n_treated"]
    out["treated_share"] = out["n_treated"] / out["n_rows"]
    return {k: (float(v) if isinstance(v, float) else v) for k, v in out.items()}


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
        "p_treated_design": p_treated,
        "p_treated_observed": n_treated / n,
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_flag": bool(p < 0.001),
    }


def feature_means_by_arm(sdf: DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Mean and standard deviation of each feature by arm, plus the standardized
    mean difference ``(mean_T - mean_C) / sqrt((var_T + var_C) / 2)``.

    One aggregation over the full data; the result is small enough for pandas.
    """
    aggs = []
    for f in features:
        aggs += [F.avg(f).alias(f"{f}__mean"), F.var_samp(f).alias(f"{f}__var")]
    rows = {r["treatment"]: r.asDict() for r in sdf.groupBy("treatment").agg(*aggs).collect()}
    t, c = rows[1], rows[0]
    out = pd.DataFrame(
        {
            "feature": features,
            "mean_treated": [t[f"{f}__mean"] for f in features],
            "mean_control": [c[f"{f}__mean"] for f in features],
            "sd_treated": [t[f"{f}__var"] ** 0.5 for f in features],
            "sd_control": [c[f"{f}__var"] ** 0.5 for f in features],
        }
    )
    pooled = ((out["sd_treated"] ** 2 + out["sd_control"] ** 2) / 2) ** 0.5
    out["smd"] = (out["mean_treated"] - out["mean_control"]) / pooled
    return out


def compare_with_pandas(spark_summary: dict, pandas_summary: dict, tol: float = 1e-9) -> dict:
    """Check Spark agrees with the pandas summary on every shared key.

    Counts must match exactly; rates within ``tol`` (floating point only).
    """
    out = {}
    for key, pv in pandas_summary.items():
        if key not in spark_summary:
            continue
        sv = spark_summary[key]
        out[key] = sv == pv if isinstance(pv, int) else abs(sv - pv) <= tol
    return out


def run(
    cfg: dict | None = None,
    spark: SparkSession | None = None,
    tables_dir: str | Path = ROOT / "reports" / "tables",
) -> dict:
    """Every Day 2 check on the full data. Returns the result dict it writes."""
    cfg = cfg or load_config()
    own = spark is None
    if own:
        spark = get_spark(
            "criteo-checks", cfg["spark"]["driver_memory"], cfg["spark"]["shuffle_partitions"]
        )
    try:
        sdf = spark.read.parquet(cfg["data"]["full_parquet"]).cache()
        summary = counts_and_rates(sdf)
        means = feature_means_by_arm(sdf)
        sdf.unpersist()
    finally:
        if own:
            spark.stop()

    result = {
        "summary": summary,
        "srm": srm_test(
            summary["n_treated"], summary["n_control"], cfg["experiment"]["treated_share_design"]
        ),
        "published_checks": check_expected(summary, cfg["expected"]),
        "max_abs_smd": float(means["smd"].abs().max()),
        "features_abs_smd_over_0.1": means.loc[means["smd"].abs() > 0.1, "feature"].tolist(),
    }
    pandas_path = Path(cfg["data"]["summary_json"])
    if pandas_path.exists():
        pandas_full = json.loads(pandas_path.read_text())["full"]
        result["pandas_checks"] = compare_with_pandas(summary, pandas_full)

    pandas_path.with_name("spark_checks.json").write_text(json.dumps(result, indent=2))
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    means.to_csv(tables_dir / "feature_means_by_arm.csv", index=False, float_format="%.6g")
    result["feature_means"] = means
    return result


def main() -> None:
    result = run()
    means = result.pop("feature_means")
    print(json.dumps(result, indent=2))
    print(means.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    failed = [k for k, ok in result["published_checks"].items() if not ok]
    failed += [f"pandas:{k}" for k, ok in result.get("pandas_checks", {}).items() if not ok]
    if result["summary"]["n_exposed_control"] != 0:
        failed.append("exposure in control")
    if failed:
        raise SystemExit(f"Spark checks failed: {failed}")
    print("Spark matches published values and the pandas summary.")


if __name__ == "__main__":
    main()
