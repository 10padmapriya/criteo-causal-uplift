# Data card: Criteo Uplift Prediction v2.1 (draft)

Status: draft, updated Day 3. Numbers below come from `make data` (pandas), `make spark` (PySpark) and `make experiment` on the full file; pandas and Spark agree on every shared count and rate. Sections marked TODO fill in as later phases run.

## Source

| | |
|---|---|
| Name | Criteo Uplift Prediction Dataset, version 2.1 |
| Publisher | Criteo AI Lab |
| Reference | Diemert, Betlei, Renaudin, Amini (2018), "A Large Scale Benchmark for Uplift Modeling", AdKDD |
| Download | Hugging Face mirror `criteo/criteo-uplift`, Criteo's own link as fallback (see `config.yaml`) |
| Licence | CC BY NC SA 4.0 (non commercial use) |
| Local copies | `data/criteo_full.parquet` (full), `data/criteo_10pct.parquet` (10% simple random sample, seed 42); neither is committed |

## What a row is

One user in one of several pooled randomized ad incrementality tests. Users were randomly assigned to a treated arm (eligible to see the advertiser's ads) or a control arm (ads withheld), then followed for visits and conversions. The data has no timestamps, test identifiers or user identifiers.

## Columns

| Column | Type | Meaning | Role in this project |
|---|---|---|---|
| f0 to f11 | float | Anonymized, dense, pre treatment user features | Covariates for balance, CUPED, propensity and uplift models |
| treatment | 0/1 | Randomized assignment, 1 = treated | The treatment |
| exposure | 0/1 | User was actually shown an ad; always 0 in control | Compliance only. Never a covariate (post treatment) |
| visit | 0/1 | User visited the advertiser's site | Primary outcome |
| conversion | 0/1 | User converted | Secondary outcome |

## Headline numbers (full data)

| Quantity | Value | Published |
|---|---|---|
| Rows | 13,979,592 | 13,979,592 |
| Treated | 11,882,655 | |
| Control | 2,096,937 | |
| Treated share | 0.8500 | 0.85 |
| Visit rate | 0.04699 | 0.047 |
| Conversion rate | 0.002917 | 0.0029 |
| Exposure rate among treated | 0.0360 | |
| Exposed users in control | 0 | 0 by design |

Intent to treat effect on the full data (`reports/tables/itt_full.csv`; unpooled SE, normal 95% CIs, delta method for the relative lift, Holm across the two outcomes):

| Outcome | Control | Treated | Lift (pts) [95% CI] | Relative lift [95% CI] | Holm p | MDE (relative, 80% power) |
|---|---|---|---|---|---|---|
| Visit | 3.820% | 4.854% | +1.034 [1.006, 1.063] | 27.1% [26.2%, 28.0%] | < 1e-300 | 1.1% |
| Conversion | 0.194% | 0.309% | +0.115 [0.108, 0.122] | 59.4% [54.3%, 64.6%] | 3.2e-246 | 4.8% |

On the 10% sample the visit lift is +1.075 pts [0.984, 1.165], consistent with the full data and about three times wider, as expected from sqrt(10).

## Integrity checks

- **Sample ratio mismatch.** Chi square against the design split of 0.85/0.15: chi2 = 1.8e-06, p = 0.999. No SRM. (Against 0.50/0.50 the same counts would fail outright, which is why the null must be 0.85.)
- **One sided noncompliance holds.** No control user has `exposure = 1`.
- **No missing values; binary columns are strictly 0/1** (`src.data.validate`).
- **Spark and pandas agree** on every count exactly and every rate to 1e-9.

## Covariate balance by arm

Full tables: `reports/tables/balance_full.csv` (pandas, with variance ratios) and `reports/tables/feature_means_by_arm.csv` (Spark). The largest absolute standardized mean difference is 0.049 (f3); no feature crosses the usual 0.1 flag. Next largest: f6 (0.040), f5 (0.031).

Variance ratios (treated over control) are all inside Rubin's 0.5 to 2 band, so nothing is flagged, but several sit well above 1: f1 1.32, f5 1.22, f3 1.19. Treated users are consistently a little more spread out than control users.

Two things worth noting for later phases:

- With 14M rows, even these small differences are far outside sampling noise, so "statistically significant imbalance" is expected and uninformative. SMD, not p values, is the right yardstick here.
- Spreads differ more than means in some features: f1 has SD 0.107 in treated against 0.093 in control, f5 0.437 against 0.396. The plan says the data pools several tests; if those tests had different treated shares or populations, pooling could produce this pattern. The variance ratios above quantify it; it is a finding to mention, not a defect that invalidates the experiment.

## Known traps

- **Split is 85/15 by design.** Every SRM test uses 0.85.
- **Exposure is post treatment.** Conditioning on it compares self selected exposed users with all control users and biases the effect. It is used only to scale ITT into the effect on the exposed (ITT divided by 0.036).
- **Conversions are rare** (0.29%). Intervals on conversion are wide, so visit leads every result.
- **Features are anonymized.** Segments are described by predicted uplift score, never by named traits.
- **Pooled tests, no test identifier.** Heterogeneity across tests cannot be separated from heterogeneity across users.
- **No time dimension.** Difference in differences and sequential testing on real timestamps are not possible; the peeking analysis is a simulation.

## Intended use and limits

Intended: benchmarking experiment analysis, observational estimators against an experimental truth, and uplift models. Not suitable for drawing conclusions about specific advertisers, user groups or real ad prices; any cost per impression used in Phase 4 is a labeled assumption, not a Criteo fact.

## TODO as phases land

- Day 4: CUPED CI shrinkage; effect on the exposed with bootstrap CI; Spark vs pandas cross validation of ITT.
- Day 5 onward: describe the confounded samples built from this data.
