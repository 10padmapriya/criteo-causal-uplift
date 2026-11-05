"""Download Criteo Uplift v2.1, convert it to Parquet, draw the 10% sample.

Run once with ``make data`` (``python -m src.data``). Every step is idempotent:
existing files are reused, so a rerun only redoes what is missing.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

FEATURES = [f"f{i}" for i in range(12)]
BINARY_COLS = ["treatment", "conversion", "visit", "exposure"]
COLUMNS = FEATURES + BINARY_COLS

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    """Read config.yaml and resolve data paths against the repo root."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("raw_csv", "full_parquet", "sample_parquet", "summary_json"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    return cfg


def download(urls: list[str], dest: str | Path, chunk: int = 1 << 20) -> Path:
    """Stream the first reachable URL to ``dest``. Skips if ``dest`` exists.

    Writes to ``dest.part`` and renames on success so an interrupted download
    never looks complete.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "criteo-causal-uplift"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out, chunk)
            part.rename(dest)
            return dest
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
    raise RuntimeError("All download URLs failed:\n" + "\n".join(errors))


def arrow_schema() -> pa.Schema:
    """Features as float64, the four binary columns as int8."""
    return pa.schema([(c, pa.float64()) for c in FEATURES] + [(c, pa.int8()) for c in BINARY_COLS])


def csv_to_parquet(csv_path: str | Path, parquet_path: str | Path) -> Path:
    """Stream a (gzipped) CSV into Parquet batch by batch, with a fixed schema.

    Streaming keeps memory flat, so the 14M row file converts on a laptop.
    """
    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        return parquet_path
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = arrow_schema()
    convert = pacsv.ConvertOptions(column_types={f.name: f.type for f in schema})
    reader = pacsv.open_csv(csv_path, convert_options=convert)
    tmp = parquet_path.with_suffix(".parquet.part")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_table(pa.Table.from_batches([batch]).select(schema.names).cast(schema))
    tmp.rename(parquet_path)
    return parquet_path


def validate(df: pd.DataFrame) -> None:
    """Raise if the frame breaks the dataset's structural rules."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for col in BINARY_COLS:
        bad = set(pd.unique(df[col])) - {0, 1}
        if bad:
            raise ValueError(f"{col} has non binary values {sorted(bad)}")
    if (df.loc[df["treatment"] == 0, "exposure"] != 0).any():
        raise ValueError("exposure must be 0 for every control user")
    if df[COLUMNS].isna().any().any():
        raise ValueError("unexpected missing values")


def summarize(df: pd.DataFrame) -> dict:
    """Headline counts and rates used to check the download against Criteo's paper."""
    treated = df["treatment"] == 1
    return {
        "n_rows": int(len(df)),
        "n_treated": int(treated.sum()),
        "n_control": int((~treated).sum()),
        "treated_share": float(treated.mean()),
        "visit_rate": float(df["visit"].mean()),
        "conversion_rate": float(df["conversion"].mean()),
        "exposure_rate_treated": float(df.loc[treated, "exposure"].mean()),
    }


def check_expected(summary: dict, expected: dict) -> dict[str, bool]:
    """Compare a summary with the published values. Row count must match exactly."""
    tol = expected["tolerance"]
    return {
        "n_rows": summary["n_rows"] == expected["n_rows"],
        "treated_share": abs(summary["treated_share"] - expected["treated_share"]) <= 0.005,
        "visit_rate": abs(summary["visit_rate"] - expected["visit_rate"]) <= tol,
        "conversion_rate": abs(summary["conversion_rate"] - expected["conversion_rate"]) <= tol,
    }


def make_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Simple random sample of rows.

    Deliberately not stratified on treatment: a stratified sample would force the
    85/15 split and make an SRM test on the sample meaningless.
    """
    rng = np.random.default_rng(seed)
    keep = rng.random(len(df)) < frac
    return df.loc[keep].reset_index(drop=True)


def load(
    which: str = "sample", columns: list[str] | None = None, cfg: dict | None = None
) -> pd.DataFrame:
    """Load the ``"sample"`` (10%) or ``"full"`` Parquet as a DataFrame."""
    cfg = cfg or load_config()
    key = {"sample": "sample_parquet", "full": "full_parquet"}[which]
    return pd.read_parquet(cfg["data"][key], columns=columns)


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    print(f"[1/4] download -> {d['raw_csv']}")
    download(d["urls"], d["raw_csv"])
    print(f"[2/4] csv -> {d['full_parquet']}")
    csv_to_parquet(d["raw_csv"], d["full_parquet"])

    print("[3/4] validate and summarize full data")
    full = pd.read_parquet(d["full_parquet"])
    validate(full)
    summary = {"full": summarize(full)}
    checks = check_expected(summary["full"], cfg["expected"])

    print(f"[4/4] {d['sample_frac']:.0%} sample -> {d['sample_parquet']}")
    if not Path(d["sample_parquet"]).exists():
        make_sample(full, d["sample_frac"], cfg["seed"]).to_parquet(
            d["sample_parquet"], compression="zstd", index=False
        )
    summary["sample"] = summarize(pd.read_parquet(d["sample_parquet"]))
    summary["checks"] = checks

    Path(d["summary_json"]).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise SystemExit(f"Published value check failed: {checks}")
    print("Data matches published values.")


if __name__ == "__main__":
    main()
