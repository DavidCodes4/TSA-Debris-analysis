# Space Debris Risk Prediction — Time Series Analysis Edition

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.13-red?logo=keras)](https://keras.io)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-lightblue)](https://www.statsmodels.org)
[![sgp4](https://img.shields.io/badge/SGP4-2.25-green)](https://pypi.org/project/sgp4)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

---

## Overview

This project builds a **local time-series analysis pipeline** for space debris trajectory forecasting and collision risk assessment. It processes Two-Line Element (TLE) data from historical archives, propagates orbital positions using SGP4, trains multiple forecasting models ranging from classical VAR to a custom PatchTST Transformer, screens all tracked object pairs for close approaches, and feeds results into a React dashboard via a Flask REST API.

The system tracks approximately 22,000 active objects in Earth orbit — satellites, rocket bodies, and debris — and forecasts their positions up to 14 hours ahead, flagging any pair that approaches within 50 km as a potential conjunction event.

---

## Problem Statement

There are over 30,000 tracked objects in orbit. When two objects come within 50 km of each other, it is a close approach. Below 1 km, it becomes a collision risk requiring operator intervention. The challenge is computational: for 15,000 active objects, a naive pair-screening approach requires evaluating 112 million distance calculations every 5 minutes.

Additionally, forecasting orbital trajectories is non-trivial. Orbital motion is periodic and nonlinear. A satellite's position along its X-axis oscillates sinusoidally with a period equal to its orbital period (approximately 90 minutes for LEO objects). Classical models struggle to capture this periodicity across a multi-hour forecast horizon.

---

## Data

### Source
Historical TLE archives from Space-Track.org stored as plain-text files, covering 2004 through 2025 (26 files, with `tle.txt` being the primary source at 2.6 GB).

### Format
Three-line TLE records: one name line followed by two data lines encoding the Keplerian orbital elements, drag coefficients, and the reference epoch.

### Processing
Rather than interpolating between raw TLE epochs (which produces piecewise-linear artifacts), this pipeline propagates each TLE forward in time using the `sgp4` library at fixed 5-minute intervals. This gives physically accurate sinusoidal position traces reflecting real Keplerian dynamics including atmospheric drag effects encoded in the B* term.

```
Raw TLE text
    │
    ▼  Parse → one record per NORAD ID (latest epoch wins)
Unique TLE per object  (~22,253 objects in tle2024.txt)
    │
    ▼  SGP4 propagation  (5-min steps, 72h horizon per object)
ECI Cartesian state vectors  (x, y, z km  |  vx, vy, vz km/s)
    │
    ▼  Quality filter  (altitude 150–100,000 km, span ≥ 12h)
Clean time-series  (200 objects, 173,000 timesteps)
    │
    ▼  First difference  (d=1 to achieve stationarity)
Stationary differenced series  → model input
```

**Sample object:** NORAD 25544 (International Space Station), 860 timesteps, altitude ~420 km.

---

## Orbital Mechanics Background

All positions are computed in the **ECI TEME frame** (Earth-Centered Inertial, True Equator Mean Equinox). The three axes X, Y, Z represent distances in kilometers from Earth's center.

Orbital altitude is derived as:

```
altitude = sqrt(x² + y² + z²) - 6371 km
```

Orbital period follows Kepler's Third Law:

```
T = 2π × sqrt(r³ / GM)   where GM = 398600.4418 km³/s²
```

For the ISS at 420 km altitude, this gives T ≈ 92 minutes = approximately 18.4 five-minute timesteps per orbit. This periodicity is the fundamental signal that all forecasting models must capture.

---

## Stationarity Analysis

Before model selection, the raw position series is tested for stationarity:

| Series | ADF p-value | KPSS p-value | Verdict |
|---|---|---|---|
| Raw POS_X | 0.18 | < 0.05 | Non-stationary |
| Raw POS_Y | 0.22 | < 0.05 | Non-stationary |
| Raw POS_Z | 0.09 | < 0.05 | Non-stationary |
| d=1 ΔPOS_X | < 0.001 | 0.10 | Stationary |
| d=1 ΔPOS_Y | < 0.001 | 0.10 | Stationary |
| d=1 ΔPOS_Z | < 0.001 | 0.10 | Stationary |

All series require exactly one order of differencing (d = 1), confirming I(1) behaviour — consistent with orbital motion where velocity (not position) is the mean-reverting quantity.

---

## Model Selection — Complete Analysis

### Why Multiple Models?

No single model class dominates orbital time-series. The signal has three distinct structural components:

1. **Periodic sinusoidal orbit** (deterministic, ~92 min period)
2. **Slow drift** due to orbital decay and atmospheric drag (trend)
3. **Noise** from TLE epoch quantisation and propagation errors

The model selection strategy matches model class to component:

| Component | Best model class | Reason |
|---|---|---|
| Periodic signal + cross-axis coupling | VAR | Captures x/y/z velocity relationships simultaneously |
| Residual nonlinear structure | LSTM / PatchTST | Handles autocorrelated residuals after linear fit |
| Full nonlinear regime switching | SETAR / LSTAR | Models ascending vs descending orbital phase |
| Smooth periodic transition | ESTAR | Models perigee (fast) vs apogee (slow) phase |

---

### Linear Models

#### AR — AutoRegressive

Univariate model on ΔPOS_X only. Grid search from AR(1) to AR(15).

**Result:** AR(10) achieves lowest AIC among univariate models. The lag-10 coefficient is significantly non-zero, indicating 50-minute memory (10 × 5 min) — approximately half the orbital period.

```
AR(10) AIC = 8527  |  RMSE = 2246 km (test set)
```

The long lag structure reflects the orbital sinusoid: to predict where the satellite is now, knowing where it was half an orbit ago is highly informative.

#### ARIMA(3,1,2)

Best ARIMA configuration by AIC. The MA(2) terms help absorb short-memory noise but do not add predictive power for the multi-step horizon.

```
ARIMA(3,1,2) AIC = 8578  |  RMSE = 2511 km
```

ARIMA is outperformed by AR(10) because the extra MA terms introduce estimation noise rather than capturing signal.

#### VAR — Vector AutoRegression

Multivariate model on (ΔPOS_X, ΔPOS_Y, ΔPOS_Z) simultaneously. AIC selection over lags 1 to 16.

**Result:** VAR(13) is optimal. Lag 13 × 5 minutes = 65 minutes ≈ approximately two-thirds of an ISS orbital period.

```
VAR(13) AIC = 18.44  |  Training RMSE = 471 km  |  Test RMSE = 2330 km
```

The VAR residuals show Ljung-Box statistics of 113–138 (all p ≈ 0), confirming significant residual autocorrelation. This is the mathematical justification for adding a neural network correction layer on top of VAR.

**Why VAR over univariate AR?**  
In orbital mechanics, X, Y, and Z are not independent. A satellite moving in the X direction has predictable Y and Z velocities from the angular momentum vector. VAR explicitly models these cross-axis velocity relationships. The cross-equation coefficient matrix captures the rotation of the orbital plane.

---

### Nonlinear Regime-Switching Models

#### SETAR — Self-Exciting Threshold AutoRegression

SETAR divides the state space into two regimes using a hard threshold on a delay variable, fitting a separate AR model in each regime.

**Grid search:** p ∈ {3, 5, 7, 10}, d ∈ {1, 2, 3}, c over the range of observed ΔPOS_X values.

**Result:** SETAR(10, 1, c = -39.70)

```
Regime 1 (ΔPOS_X ≤ -39.70):  n = 234 obs,  σ = 57.7 km
Regime 2 (ΔPOS_X > -39.70):  n = 443 obs,  σ = 114.5 km
SETAR AIC = 6263  |  Test RMSE = 4210 km
```

The threshold at c = -39.70 km/step corresponds to the satellite moving sharply in the negative X direction — the descending node in ECI space. The σ ratio (2:1 between regimes) indicates genuinely different dynamics: near the descending node the orbit is tighter and more predictable; at other phases velocity dispersion is larger.

**Why poor out-of-sample RMSE despite low AIC?**  
The sharp threshold creates discontinuous forecasts. When the test series crosses the threshold, the forecast jumps abruptly between two AR models. This cliff effect produces large errors around regime transitions.

#### LSTAR — Logistic Smooth Transition AutoRegression

LSTAR replaces the hard threshold with a logistic transition function:

```
G(s; γ, c) = 1 / (1 + exp(-γ(s - c)))

y_t = (1 - G) × [φ₁₀ + Σ φ₁ᵢ y_{t-i}]  +  G × [φ₂₀ + Σ φ₂ᵢ y_{t-i}]  +  ε_t
```

The parameter γ controls transition sharpness. Small γ → nearly linear. Large γ → approaches SETAR.

**Result:** LSTAR(10, d=1)

```
γ = 1.1265  (moderate sharpness)
c = -0.113   (transition centre near zero ΔPOS_X)
LSTAR AIC = 6267  |  Test RMSE = 4648 km
```

At γ = 1.13, the transition is moderately sharp with the centre near zero velocity. Effectively, the model assigns one regime to negative velocity (descending) and another to positive velocity (ascending). This is physically meaningful — ascending and descending phases of the orbit do have different atmospheric drag profiles.

**Why worse than SETAR despite smoother transitions?**  
With only 682 training points, the L-BFGS-B optimizer struggles to separately identify γ, c, and 21 regression coefficients (10 lags × 2 regimes + intercepts). Some runs converge to extreme γ values (see p=7, d=1: γ ≈ 8.6 billion, effectively SETAR), indicating optimizer instability. The best run with γ = 1.13 is well-behaved but underpowered for a 172-step horizon.

#### ESTAR — Exponential Smooth Transition AutoRegression

ESTAR uses an exponential (bell-shaped) transition function:

```
G(s; γ, c) = 1 - exp(-γ(s - c)²)
```

This assigns regime 1 to observations near the transition centre (s ≈ c) and regime 2 to observations far from it in either direction. In orbital terms: one regime for the satellite near a turning point (where ΔPOS_X ≈ 0, i.e., near perigee or apogee) and another for mid-orbit (where |ΔPOS_X| is large).

**Result:** ESTAR(10, d=1)

```
γ ≈ 0.000  (flat transition — degenerate to linear)
c = 128.5
ESTAR AIC = 6314  |  Test RMSE = 2325 km
```

The optimizer consistently returns γ ≈ 0, meaning the exponential transition is essentially flat and the model degenerates to a linear AR(10). This is actually a useful finding: it means the data does not support the specific perigee/apogee regime structure that ESTAR is designed to detect. The near-linear ESTAR achieves the best RMSE among all STAR models (2325 km) because it avoids over-fitting to regime boundaries.

#### TAR-VAR — Threshold Vector AutoRegression

Extends SETAR to the multivariate case. Grid search over the same (p, d, c) space as SETAR but fitting full 3-variable VAR models in each regime.

**Result:** TAR-VAR(5, d=1, c=-39.70)

```
TAR-VAR AIC = 6456  |  Linear VAR AIC = 18.68
ΔAIC = -6437  →  Linear VAR is decisively better
Test RMSE = 4376 km
```

The enormous AIC gap confirms that TAR-VAR is grossly overfit. Each regime fits a separate VAR(5) model with 3 variables, resulting in 2 × (1 + 5×3) × 3 = 96 coefficients on only 162 and 520 observations respectively. The information criterion correctly penalises this.

---

### Deep Learning Models

#### LSTM — Long Short-Term Memory

Architecture: two stacked LSTM layers (64 → 32 units) with dropout, operating on a 60-step lookback window over scaled (ΔPOS_X, ΔPOS_Y, ΔPOS_Z).

```
Input:  (batch, 60 timesteps, 3 features)
        │
    LSTM(64, return_sequences=True, dropout=0.1)
        │
    LSTM(32, return_sequences=False, dropout=0.1)
        │
    Dense(1)  →  predicted ΔPOS_X at t+1
```

```
Parameters:  50,627
Best epoch:  5  (early stopping, patience=5)
val MSE:     0.000131
```

The LSTM is used in two roles:
1. **Standalone:** recursive multi-step forecast by feeding predicted ΔPOS_X back as input
2. **Hybrid correction:** applied to VAR residuals to absorb the autocorrelated structure that VAR leaves unexplained

**Why does LSTM stop at epoch 5?**  
With only 550 training sequences (from 687 training timesteps with 60-step lookback), the LSTM saturates quickly. The model is slightly underpowered for the amount of data. On SGP4-densified data (this pipeline), the longer sinusoidal history gives LSTM more signal to learn from compared to the notebook experiments.

#### PatchTST — Patch Time Series Transformer

Custom implementation of Nie et al. (2023) "A Time Series is Worth 64 Words" using Keras `MultiHeadAttention`. No external library required.

**Key design decision: patch length = 17 steps = 85 minutes ≈ 1 orbital period**

This is the critical insight. By setting the patch length to match the orbital period, each patch token in the Transformer corresponds to exactly one complete orbit. The self-attention mechanism then naturally learns to compare the satellite's state at the same orbital phase across multiple revolutions — capturing the periodicity explicitly in the attention weights.

```
Architecture:

Input: (batch, n_patches, 17, 3)
           │
   Reshape: (batch, n_patches, 51)   ← flatten each patch
           │
   Dense → d_model=64                ← patch embedding
           │
   + Positional Encoding (learned Embedding)
           │
   ┌───────┤  × 2 Transformer layers
   │  MultiHeadAttention(4 heads, key_dim=16)
   │  LayerNorm + Residual
   │  FFN: Dense(128, relu) → Dropout(0.1) → Dense(64)
   │  LayerNorm + Residual
   └───────┤
   GlobalAveragePooling1D            ← aggregate patch tokens
           │
   Dropout(0.1)
           │
   Dense(1)                          ← predicted ΔPOS_X at t+1

Parameters:  ~85,000
```

**Comparison to LSTM:**

| Property | LSTM | PatchTST |
|---|---|---|
| Receptive field | 60 steps (5 hrs) | 4 × 17 = 68 steps (5.7 hrs) |
| Orbital periods seen | ~3.9 orbits | ~4.0 orbits |
| Periodicity mechanism | Implicit, via hidden state | Explicit, via patch-to-patch attention |
| Training parallelism | Sequential (BPTT) | Fully parallel |
| Sensitivity to orbit phase | Weak — forgets early context | Strong — attention across phases |
| Expected advantage on SGP4 data | Baseline | Captures perigee/apogee asymmetry |

PatchTST is expected to outperform LSTM on SGP4-densified data (this pipeline's output) because the real sinusoidal orbit provides the periodic structure that attention is designed to capture. In the notebook experiments (interpolated data), both models performed similarly because the periodic signal was suppressed by interpolation artifacts.

---

### Hybrid Models

The VAR residuals contain significant autocorrelation (Ljung-Box test: LB statistic 113–138, all p ≈ 0). Rather than discarding this information, the hybrid approach uses the neural model to forecast the residual series and adds it back to the VAR forecast:

```
Hybrid forecast = VAR forecast + α × Neural correction

where α = 0.3  (blend weight, tuned to prevent over-correction)
```

Two hybrids are evaluated:
- **Hybrid VAR + LSTM:** RMSE = 2220 km (best in main notebook)
- **Hybrid VAR + PatchTST:** Expected improvement on SGP4-densified data

The blend weight α = 0.3 is conservative by design. The neural correction is trusted less than the VAR because on short training data (682 points), neural networks can introduce spurious patterns that degrade multi-step forecasts.

---

## Full Benchmark Results

### Notebook Results (interpolated TLE data — 19 raw epochs)

| Model | RMSE (km) | MAE (km) | AIC |
|---|---|---|---|
| Hybrid VAR(13) + LSTM | **2,220** | 1,890 | — |
| VAR(13) | 2,330 | 1,970 | 18.44 |
| AR(10) | 2,355 | 1,992 | 8,527 |
| ESTAR(10,1) | 2,325 | 2,096 | 6,314 |
| Naïve (persistence) | 2,448 | 2,073 | — |
| ARIMA(3,1,2) | 2,511 | 2,069 | 8,578 |
| TAR-VAR(5,1) | 4,376 | 3,315 | 6,456 |
| SETAR(10,1) | 4,210 | 3,156 | 6,263 |
| LSTAR(10,1) | 4,648 | 3,449 | 6,267 |

The high absolute RMSE (2,000+ km) is explained by the data limitation: 19 raw TLE epochs linearly interpolated into 860 timesteps produces piecewise-linear artifacts. The real orbital sinusoid is suppressed. This inflates all model errors equally.

### Pipeline Results (SGP4-densified data — this codebase)

With real SGP4 propagation at 5-minute steps, each object has genuine sinusoidal trajectory data. Expected ranking:

```
PatchTST ≈ Hybrid VAR + PatchTST  >  Hybrid VAR + LSTM  >  VAR  >  AR  >  Naïve
```

ESTAR and SETAR are expected to remain competitive (or improve) because the real perigee/apogee speed asymmetry now exists in the data for the transition functions to detect.

---

## Residual Diagnostics Summary

| Model | Residual σ (km) | ADF (stationary) | Ljung-Box (white noise) |
|---|---|---|---|
| AR | 124.2 | Yes | No (LB = 38.9, p ≈ 0) |
| SETAR | 95.0 | Yes | No (LB = 61.2, p ≈ 0) |
| LSTAR | 94.9 | Yes | No (LB = 65.9, p ≈ 0) |
| ESTAR | 91.1 | Yes | No (LB = 128.8, p ≈ 0) |

All models leave autocorrelated residuals — none achieve white noise. The Ljung-Box statistics increase from AR (38.9) to ESTAR (128.8), indicating that the nonlinear models, despite lower residual variance, paradoxically introduce more structured periodicity into their residuals. This is consistent with over-fitting to regime boundaries rather than capturing the underlying orbital sinusoid.

---

## Conjunction Assessment

For multi-object risk screening, all pair combinations of propagated objects are evaluated for close approaches over a 24-hour window.

### Algorithm

Naive O(n²) screening is replaced by altitude bucketing (matching the approach used in the original Spark pipeline):

```
1.  Assign each object an altitude bucket: floor(altitude / 50 km)
2.  For each object in bucket B, only compare against objects in B-1, B, B+1
3.  Compute exact 3D Euclidean distance for candidate pairs
4.  Record time of closest approach (TCA) and distance
```

This reduces the search space from O(n²) to approximately O(n × k) where k is the average number of objects per 50-km altitude band.

### Risk Classification

| Risk Level | Distance Threshold | Action |
|---|---|---|
| CRITICAL | < 1 km | Immediate operator alert |
| HIGH | 1 – 5 km | Manoeuvre assessment required |
| MEDIUM | 5 – 20 km | Increased monitoring |
| LOW | 20 – 50 km | Watch list |

Collision probability uses a Gaussian miss-distance approximation (Chan 2008):

```
P ≈ (π r²) / (2π σ²)

where r = combined hard-body radius (~10 m)
      σ = max(distance / 3, r)
```

---

## Pipeline Architecture

```
TLE Archive (data/raw/*.txt)
        │
        ▼  01_data_pipeline.py
SGP4 Densification  (5-min steps, 72h per object)
→  Output/ts_df.parquet  (200 objects, ~173,000 timesteps)
        │
        ▼  02_train_models.py
Model Training
├── VAR(p)   — statsmodels, AIC lag selection
├── LSTM     — Keras stacked LSTM, early stopping
└── PatchTST — Custom Keras Transformer, patch_len=17
→  Output/models/  (var_model.pkl, lstm_model.keras, patchtst_model.keras)
        │
        ▼  03_evaluate_models.py
Benchmark Evaluation  (recursive multi-step, 172 steps = 14 hrs)
├── Naïve / AR / ARIMA / VAR / LSTM / PatchTST
├── Hybrid VAR+LSTM
└── Hybrid VAR+PatchTST
→  Output/forecast_results.json
→  Output/benchmark_table.csv
        │
        ▼  04_conjunction_assessment.py
Multi-Object Risk Screening  (all 200-object pairs, 24h window)
→  Output/collision_alerts_<timestamp>/part-00000.csv
→  Output/conjunction_report.json
        │
        ▼  05_update_dashboard.py
Dashboard Consolidation
→  Output/dashboard_state.json   (consumed by Flask API)
→  Output/space_debris_catalog.csv
        │
        ▼  dashboard_api.py  (Flask, port 5050)
REST API
├── GET /api/dashboard/stats
├── GET /api/collisions
├── GET /api/collisions/globe
├── GET /api/tsa/state         ← new: full pipeline state
├── GET /api/tsa/forecast      ← new: per-model forecasts
└── GET /api/tsa/conjunctions  ← new: conjunction report
        │
        ▼  dashboard/  (React + Vite, port 5173)
Interactive Dashboard
```

---

## Quick Start

**Prerequisites:** Python 3.13, Node.js 18+, `make`

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv
make install

# 2. Run the full pipeline  (~15–20 min, dominated by SGP4 propagation)
make pipeline

# 3. Start the API server
make api          # Flask on http://localhost:5050

# 4. Start the dashboard  (separate terminal)
make dev          # React on http://localhost:5173
```

**Common workflows:**

```bash
make dry-run      # preview all steps without executing
make step1        # only re-run data pipeline (new TLE file)
make step2        # only re-train models
make refresh      # steps 3+4+5 — skip training, re-evaluate saved models
make clean        # remove all generated artefacts (preserves raw TLE data)
```

**Run individual scripts directly:**

```bash
# Full pipeline with custom parameters
.venv/bin/python scripts/models/run_pipeline.py \
    --tle-file tle2025_.txt \
    --top-n 500 \
    --propagate-hrs 96 \
    --sample-id 25544

# Only conjunction screening with tighter threshold
.venv/bin/python scripts/models/04_conjunction_assessment.py \
    --screen-km 20 \
    --screen-hours 48
```

---

## API Reference

All endpoints are served by `dashboard_api.py` on port 5050.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service health, HDFS connectivity |
| GET | `/api/dashboard/stats` | Aggregated risk statistics |
| GET | `/api/collisions` | Paginated collision alerts (filterable by risk level) |
| GET | `/api/collisions/globe` | Globe-view positions for up to 500 alerts |
| GET | `/api/collisions/high-risk` | CRITICAL and HIGH alerts only |
| GET | `/api/collisions/frequency` | Most frequently appearing object pairs |
| GET | `/api/debris` | Debris catalog with orbital parameters |
| GET | `/api/ml/metrics` | Model training metrics |
| GET | `/api/tsa/state` | Full TSA pipeline state (generated by step 5) |
| GET | `/api/tsa/forecast` | Per-model forecast time-series |
| GET | `/api/tsa/conjunctions` | Conjunction report, filterable by `?risk_level=HIGH` |

---

## Notebooks

Two Jupyter notebooks document the analysis and are self-contained (all data loading inline):

| Notebook | Purpose |
|---|---|
| `notebooks/space_debris_ts_analysis.ipynb` | Full 12-section analysis: TLE parsing → SGP4 → EDA → stationarity → AR/ARIMA/VAR selection → LSTM → hybrid forecasting → conjunction assessment |
| `notebooks/tar_star_analysis.ipynb` | Regime-switching models: SETAR, LSTAR, ESTAR, TAR-VAR — with grid search, diagnostics, and full benchmark against linear models |

To run with the project's virtual environment:

```bash
.venv/bin/python -m ipykernel install --user \
    --name=tsa-venv --display-name "Python (.venv TSA)"
```

Then select "Python (.venv TSA)" as the kernel in Jupyter.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `sgp4` | 2.25 | SGP4 orbital propagation |
| `tensorflow` / `keras` | 2.21 / 3.13 | LSTM, PatchTST |
| `statsmodels` | 0.14 | VAR, ARIMA, AR, ADF/KPSS tests, Ljung-Box |
| `pandas` | 3.0 | Data wrangling |
| `numpy` | 2.4 | Numerical arrays |
| `scipy` | 1.17 | SETAR/LSTAR/ESTAR optimization (L-BFGS-B) |
| `scikit-learn` | 1.8 | MinMaxScaler, metrics |
| `pyarrow` | 23.0 | Parquet read/write |
| `flask` / `flask-cors` | 3.1 / 6.0 | REST API |
| `matplotlib` / `seaborn` | 3.10 / 0.13 | Visualisation |

---

## Key References

- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023.
- Hoots, F. R. & Roehrich, R. L. (1980). *Spacetrack Report No. 3: Models for Propagation of NORAD Element Sets.* (SGP4 specification)
- Chan, F. K. (2008). *Spacecraft Collision Probability.* Aerospace Press.
- Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach.* Oxford University Press. (TAR/SETAR)
- Teräsvirta, T. (1994). *Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models.* JASA. (LSTAR/ESTAR)
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis.* Springer. (VAR)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
