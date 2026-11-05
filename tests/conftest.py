"""Shared synthetic fixtures. Real data never enters the test suite."""

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES


@pytest.fixture
def criteo_like() -> pd.DataFrame:
    """Small frame with Criteo's schema, an 85/15 split and one sided noncompliance."""
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["treatment"] = (rng.random(n) < 0.85).astype("int8")
    df["exposure"] = ((rng.random(n) < 0.04) & (df["treatment"] == 1)).astype("int8")
    df["visit"] = (rng.random(n) < 0.04 + 0.01 * df["treatment"]).astype("int8")
    df["conversion"] = ((rng.random(n) < 0.06) & (df["visit"] == 1)).astype("int8")
    return df
