"""Proves Java 17 and PySpark work, and Spark agrees with pandas on a toy frame."""

import pytest

pyspark = pytest.importorskip("pyspark")

from src.spark_checks import get_spark  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = get_spark("test", driver_memory="1g", shuffle_partitions=2)
    yield s
    s.stop()


def test_spark_matches_pandas(spark, criteo_like):
    sdf = spark.createDataFrame(criteo_like)
    assert sdf.count() == len(criteo_like)
    visit = sdf.agg({"visit": "avg"}).collect()[0][0]
    assert visit == pytest.approx(criteo_like["visit"].mean())
