# Space Debris TSA — Model Scripts

## Overview

```
scripts/models/
├── 01_data_pipeline.py        TLE → SGP4 densification → ts_df.parquet
├── 02_train_models.py         VAR + LSTM + PatchTST → saved models
├── 03_evaluate_models.py      Benchmark all models → forecast_results.json
├── 04_conjunction_assessment.py  Multi-object risk screening → collision CSV
├── 05_update_dashboard.py     Consolidate → dashboard_state.json + catalog.csv
└── run_pipeline.py            End-to-end orchestrator
```

The output of Step 1 (`Output/ts_df.parquet`) is also the data source for:
- `notebooks/trajectory_visualisation.ipynb` — 8-section Plotly notebook (3D ECI orbits, ground tracks, altitude profiles, animated trajectories)
- `dashboard/src/pages/TrajectoryPage.tsx` — animated ground-track page served via the trajectory API endpoints

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the full pipeline (all 5 steps, ~10-20 min first run)
cd scripts/models
python run_pipeline.py

# Run only specific steps (e.g. skip training, just re-run evaluation + dashboard)
python run_pipeline.py --steps 3 4 5

# Start the Flask API (serves the dashboard)
cd ../..
python dashboard_api.py

# Start the React dashboard (in a separate terminal)
cd dashboard && npm run dev
```

## Step-by-step

### Step 1 — Data Pipeline
```bash
python 01_data_pipeline.py --tle-file tle2024.txt --top-n 200 --propagate-hrs 72
```
**Key difference from notebook**: instead of linear interpolation between raw TLE epochs,
this script uses `sgp4.api.Satrec.sgp4()` to propagate each TLE forward at 5-minute steps,
giving physically accurate sinusoidal orbits rather than piecewise-linear artifacts.

Output: `Output/ts_df.parquet`, `Output/metadata.json`

### Step 2 — Model Training
```bash
python 02_train_models.py --sample-id 25544
```
Trains three models on NORAD 25544 (ISS) or any object in `ts_df`:

| Model     | Architecture                       | Why                                    |
|-----------|------------------------------------|----------------------------------------|
| VAR(p)    | Vector AutoRegression, AIC lag     | Strong linear multivariate baseline    |
| LSTM      | 2× stacked LSTM (64→32 units)      | Captures nonlinear residual structure  |
| PatchTST  | Transformer with 17-step patches   | One patch ≈ one orbital period (85 min)|

**Why PatchTST over plain LSTM?**
- LSTM sees data sequentially and forgets distant context
- PatchTST tokenises the orbit into patches (each ≈ one full ISS orbit)
  and applies global self-attention, directly encoding the periodic orbital signal
- No extra library needed — built entirely with `keras.layers.MultiHeadAttention`

Output: `Output/models/lstm_model.keras`, `Output/models/patchtst_model.keras`,
        `Output/models/var_model.pkl`, `Output/models/train_config.json`

### Step 3 — Evaluation
```bash
python 03_evaluate_models.py
```
Runs recursive multi-step forecasts for all models + two hybrid combinations:
- Hybrid VAR+LSTM   = VAR forecast + 0.3 × LSTM correction
- Hybrid VAR+PatchTST = VAR forecast + 0.3 × PatchTST correction

Output: `Output/forecast_results.json`, `Output/benchmark_table.csv`

### Step 4 — Conjunction Assessment
```bash
python 04_conjunction_assessment.py --screen-km 50 --screen-hours 24
```
Screens all object pairs in the propagated time-series for close approaches.
Uses NASA/ESA standard risk thresholds (CRITICAL <1 km, HIGH <5 km, MEDIUM <20 km).
Writes collision alerts in the exact format expected by `dashboard_api.py`.

Output: `Output/collision_alerts_<timestamp>/part-00000.csv`,
        `Output/conjunction_report.json`

### Step 5 — Dashboard Update
```bash
python 05_update_dashboard.py
```
Consolidates all pipeline outputs into `Output/dashboard_state.json` and rebuilds
the debris catalog CSV. The Flask API reads these files directly.

New API endpoints (added to `dashboard_api.py`):
- `GET /api/tsa/state`              — full pipeline state
- `GET /api/tsa/forecast`           — per-model forecast time-series
- `GET /api/tsa/conjunctions`       — conjunction report (filterable by risk_level)
- `GET /api/debris/trajectory`      — 72-hour ECI + geodetic trajectory for one object
- `GET /api/debris/trajectories/all`— downsampled ground-track lat/lon for all objects

## Model Architecture: PatchTST

```
Input: (batch, n_patches, patch_len, n_features)
                 │
         Linear Projection → d_model=64
                 │
         + Positional Encoding (learned)
                 │
      ┌──────────┤ × 2 layers
      │  MultiHeadAttention (4 heads)
      │  LayerNorm + Residual
      │  FFN (128 → 64)
      │  LayerNorm + Residual
      └──────────┤
      GlobalAveragePooling1D
                 │
          Dense → horizon=1
```

At 5-min steps with `patch_len=17`:
- Each patch covers 85 minutes ≈ exactly 1 ISS orbital period
- The model learns orbital-phase-aware attention weights
- This is the key advantage over LSTM for periodic orbital data
