import json

import numpy as np
import pandas as pd
import pytest

from src import data


def test_csv_to_parquet_roundtrip(tmp_path, criteo_like):
    # Criteo's CSV column order, gzipped like the real download
    csv = tmp_path / "raw.csv.gz"
    criteo_like[data.FEATURES + ["treatment", "conversion", "visit", "exposure"]].to_csv(
        csv, index=False, compression="gzip"
    )
    out = data.csv_to_parquet(csv, tmp_path / "full.parquet")
    back = pd.read_parquet(out)
    assert list(back.columns) == data.FEATURES + data.BINARY_COLS
    assert (back[data.BINARY_COLS].dtypes == "int8").all()
    pd.testing.assert_frame_equal(back, criteo_like[back.columns], check_dtype=False)


def test_csv_to_parquet_skips_existing(tmp_path):
    target = tmp_path / "full.parquet"
    target.write_text("already here")
    assert data.csv_to_parquet(tmp_path / "missing.csv.gz", target) == target


def test_download_uses_fallback_and_skips_existing(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "raw" / "file.bin"
    urls = [(tmp_path / "nope.bin").as_uri(), src.as_uri()]
    assert data.download(urls, dest).read_bytes() == b"payload"
    assert not dest.with_suffix(".bin.part").exists()
    src.unlink()  # second call must not touch the network
    assert data.download(urls, dest).read_bytes() == b"payload"


def test_download_raises_when_all_fail(tmp_path):
    with pytest.raises(RuntimeError, match="All download URLs failed"):
        data.download([(tmp_path / "nope").as_uri()], tmp_path / "x")


def test_validate_accepts_clean_frame(criteo_like):
    data.validate(criteo_like)


@pytest.mark.parametrize(
    "corrupt, msg",
    [
        (lambda d: d.drop(columns="f3"), "missing columns"),
        (lambda d: d.assign(visit=d["visit"] * 2), "non binary"),
        (lambda d: d.assign(exposure=1), "exposure must be 0"),
        (lambda d: d.assign(f0=np.nan), "missing values"),
    ],
)
def test_validate_rejects(criteo_like, corrupt, msg):
    with pytest.raises(ValueError, match=msg):
        data.validate(corrupt(criteo_like.copy()))


def test_summarize_matches_hand_computation(criteo_like):
    s = data.summarize(criteo_like)
    t = criteo_like["treatment"] == 1
    assert s["n_rows"] == len(criteo_like) == s["n_treated"] + s["n_control"]
    assert s["treated_share"] == pytest.approx(t.mean())
    assert s["visit_rate"] == pytest.approx(criteo_like["visit"].mean())
    assert s["exposure_rate_treated"] == pytest.approx(criteo_like.loc[t, "exposure"].mean())
    json.dumps(s)  # plain Python types only


def test_check_expected():
    exp = {
        "n_rows": 100,
        "treated_share": 0.85,
        "visit_rate": 0.047,
        "conversion_rate": 0.0029,
        "tolerance": 0.0005,
    }
    good = {"n_rows": 100, "treated_share": 0.8500, "visit_rate": 0.0470, "conversion_rate": 0.0029}
    assert all(data.check_expected(good, exp).values())
    bad = dict(good, n_rows=99, visit_rate=0.05)
    res = data.check_expected(bad, exp)
    assert not res["n_rows"] and not res["visit_rate"] and res["conversion_rate"]


def test_make_sample_is_random_and_reproducible(criteo_like):
    a = data.make_sample(criteo_like, 0.1, seed=42)
    b = data.make_sample(criteo_like, 0.1, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == pytest.approx(0.1 * len(criteo_like), rel=0.1)
    # not stratified, but the treated share should still be close to 0.85
    assert a["treatment"].mean() == pytest.approx(0.85, abs=0.03)


def test_load_config_resolves_paths():
    cfg = data.load_config()
    assert cfg["experiment"]["treated_share_design"] == 0.85
    assert cfg["data"]["full_parquet"].startswith(str(data.ROOT))
