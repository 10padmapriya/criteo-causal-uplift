"""Full data checks in PySpark: counts, rates, SRM, feature means, ITT (Day 2)."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


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
