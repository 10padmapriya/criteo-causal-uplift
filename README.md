# Causal Inference and Uplift Modeling on Advertising Data

I knew the true effect from a randomized test, broke the randomization on purpose, and showed which observational methods got the answer back.

Data: [Criteo Uplift Prediction v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), 13,979,592 users from randomized ad incrementality tests (85% treated, 15% control).

> Work in progress. Results tables, charts and the dashboard link land here as each phase finishes.

## Quickstart (Mac)

```bash
brew install python@3.11 openjdk@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # add to ~/.zshrc

make setup   # .venv with pinned requirements
make java    # confirms Java 17 for PySpark
make data    # downloads ~311 MB, writes data/criteo_full.parquet and data/criteo_10pct.parquet
make test    # pytest on synthetic data with known effects
make all     # lint, test, data, and later every phase
```

`make data` checks the full file against the published values (13,979,592 rows, treated share 0.85, visit 0.047, conversion 0.0029) and writes them to `data/summary.json`.

Note: `sklift.datasets.fetch_criteo` points at an S3 bucket that now returns 404, so `src/data.py` downloads from Criteo's Hugging Face mirror, with Criteo's own link as a fallback.

## Layout

```
src/
  data.py           download, Parquet, 10% sample, published value checks
  spark_checks.py   full data counts, rates, SRM, ITT in PySpark
  experiment.py     SRM, balance, ITT, MDE, CUPED, effect on exposed, peeking sim
  observational.py  confounded sampler, true ATT, naive, RA, PSM, IPW, AIPW, DML
  uplift.py         S/T/X learners, causal forest, Qini, AUUC, GATES, calibration
  policy.py         targeting curve, cost cutoff, policy tree
tests/              synthetic data with a known effect, one test file per module
notebooks/          01_experiment, 02_observational, 03_uplift (call src/ only)
app/                Streamlit ROI dashboard
reports/            memo, data card, interview prep
config.yaml         seeds, paths, sample sizes
```

## Dataset rules

- Sample ratio mismatch is tested against 0.85, not 0.50.
- `exposure` happens after assignment, so it is never a covariate. It is used only to scale ITT into the effect on the exposed.
- `visit` (4.7%) is the primary outcome; `conversion` (0.29%) is secondary.
- Development runs on the 10% sample; final numbers run on the full data.
