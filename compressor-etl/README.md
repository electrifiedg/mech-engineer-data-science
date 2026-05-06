# Compressor Performance ETL

## Business Context

Manufacturers of variable-speed reciprocating compressors run acceptance tests across a range of operating conditions before shipment. Each unit is exercised at multiple combinations of evaporating temperature (T_e) and condensing temperature (T_c), and the measured Capacity and Power are compared against an expected design curve. Units whose performance deviates significantly from the design curve may indicate assembly defects, sensor miscalibration, or refrigerant charge issues — and should be held for inspection before leaving the facility. This pipeline automates that comparison at scale.

## Pipeline Overview

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Ingest | `src/ingest.py` | `compressor_data.xlsx` | `data/compressor_clean.parquet` |
| 2. Fit | `src/fit_curve.py` | `compressor_clean.parquet` | `outputs/*.pkl` |
| 3. Flag | `src/flag_deviation.py` | parquet + pkl files | `outputs/flagged_results.csv` |

**Ingest** reads the raw Excel file, keeps the four relevant columns (`T_e`, `T_c`, `Capacity`, `Power`), enforces numeric types, validates each column against engineering range limits, and writes a clean Parquet file.

**Fit** loads the clean data, expands `[T_e, T_c]` into a 3rd-degree polynomial feature matrix, and fits two `LinearRegression` models — one for Capacity and one for Power. The fitted transformer and both models are serialized with `joblib`.

**Flag** scores every row by computing predictions from the design curve, calculates residuals, and raises a flag on any row where the absolute residual exceeds 2 standard deviations. Results including all intermediate columns are written to CSV.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run each stage in order
python src/ingest.py
python src/fit_curve.py
python src/flag_deviation.py
```

The raw Excel file is included in `data/`. Stage 1 will skip the download if the file is already present.

## Output

`outputs/flagged_results.csv` contains all rows from the clean dataset plus these additional columns:

| Column | Description |
|---|---|
| `capacity_pred` | Design-curve prediction for Capacity at this (T_e, T_c) |
| `power_pred` | Design-curve prediction for Power at this (T_e, T_c) |
| `capacity_residual` | Measured Capacity minus predicted Capacity |
| `power_residual` | Measured Power minus predicted Power |
| `capacity_flag` | `True` if \|capacity_residual\| > 2σ of all Capacity residuals |
| `power_flag` | `True` if \|power_residual\| > 2σ of all Power residuals |
| `any_flag` | `True` if either flag is raised |

**Operationally**, a flagged row means the unit's measured performance deviates from the polynomial design curve by more than 2 standard deviations at that operating condition. These units should be pulled for manual inspection.

## Data Source

Dataset: Purdue ME-239 "Introduction to Data Science for Mechanical Engineers" course dataset (Prof. Bilionis). 65 rows of compressor test data varying T_e and T_c.

Source URL: `https://raw.githubusercontent.com/PurdueMechanicalEngineering/me-239-intro-to-data-science/master/data/compressor_data.xlsx`

Note: This prototype fits the design curve to the full dataset and flags statistical outliers within that same dataset. It does not include a separate population of known-degraded units for supervised anomaly detection — that extension is left for a future iteration.
