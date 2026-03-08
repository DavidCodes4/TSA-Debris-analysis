"""
02_train_models.py
====================
Train three complementary forecasting models on the SGP4-densified
time-series produced by 01_data_pipeline.py.

Models
------
1. VAR(p)         — statsmodels VARModel  (linear multivariate baseline)
2. LSTM            — Keras stacked LSTM   (nonlinear sequence model)
3. PatchTST        — built with Keras/TF  (Transformer with patched input)
                     Inspired by: Nie et al. 2023 "A Time Series is Worth
                     64 Words" — implemented here from scratch using
                     tf.keras so no extra library is needed.

Why PatchTST over plain LSTM?
------------------------------
| Property           | LSTM           | PatchTST               |
|--------------------|----------------|------------------------|
| Receptive field    | sequential,     | global via attention   |
|                    | forgets far past|                        |
| Speed              | slow (BPTT)    | parallelisable         |
| Orbital periodicity| weak (13 lags) | patch captures 1 orbit |
| Overfitting risk   | moderate       | lower (patch=local avg)|

A 5-min step × 17-step patch ≈ 85 min ≈ 1 ISS orbit → PatchTST natively
sees one full orbit per token, making it ideal for this domain.

Outputs (saved to OUTPUT_DIR/models/):
  var_model.pkl          — statsmodels VAR fitted result (pickle)
  lstm_model.keras       — Keras saved model
  patchtst_model.keras   — Keras saved model
  scaler_X.pkl           — MinMaxScaler for input features
  scaler_y.pkl           — MinMaxScaler for target (POS_X)
  train_config.json      — hyperparameters and metrics
"""

import os, json, pickle, logging, argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
import keras
from keras import layers

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path(__file__).resolve().parents[2] / "Output"
MODELS_DIR    = OUTPUT_DIR / "models"
SAMPLE_ID     = 25544          # ISS — change to any NORAD ID in ts_df
FORECAST_STEPS = 172           # steps to forecast (≈14 hrs)
TEST_FRAC     = 0.2

# VAR
VAR_MAX_LAGS  = 16

# LSTM
LSTM_LOOKBACK = 60             # input window (steps)
LSTM_UNITS    = [64, 32]
LSTM_DROPOUT  = 0.1
LSTM_EPOCHS   = 30
LSTM_BATCH    = 32

# PatchTST
PATCH_LEN     = 17             # ~1 orbital period at 5-min steps (85 min)
PATCH_STRIDE  = 8
D_MODEL       = 64
N_HEADS       = 4
N_LAYERS      = 2
FFN_DIM       = 128
PTST_DROPOUT  = 0.1
PTST_EPOCHS   = 30
PTST_BATCH    = 32


# ── Sequence builders ─────────────────────────────────────────────────────────

def _make_sequences(data: np.ndarray, lookback: int, horizon: int = 1):
    """Sliding-window sequences. data shape: (T, F)."""
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])
        y.append(data[i : i + horizon, 0])   # target = first column (x)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _make_patch_sequences(data: np.ndarray, patch_len: int,
                          patch_stride: int, horizon: int = 1):
    """
    Patched input for PatchTST.
    Returns:
      X_patches : (N, n_patches, patch_len, n_features)
      y         : (N, horizon)
    """
    n_features = data.shape[1]
    lookback   = patch_len + (patch_len - patch_stride) * (
        (patch_len - 1) // patch_stride)   # ensure integer number of patches
    # Simpler: fixed lookback = 17 patches × stride
    lookback = patch_len * 4               # 4 patches → 68 steps ≈ 340 min

    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        window = data[i - lookback : i]    # (lookback, n_features)
        # Extract patches with stride
        patches = []
        start = 0
        while start + patch_len <= lookback:
            patches.append(window[start : start + patch_len])
            start += patch_stride
        X.append(np.stack(patches))        # (n_patches, patch_len, n_features)
        y.append(data[i : i + horizon, 0])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# ── Model builders ────────────────────────────────────────────────────────────

def _build_lstm(lookback: int, n_features: int, horizon: int,
                units: list, dropout: float) -> keras.Model:
    inp = keras.Input(shape=(lookback, n_features))
    x = inp
    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = layers.LSTM(u, return_sequences=return_seq, dropout=dropout)(x)
    x = layers.Dense(horizon)(x)
    model = keras.Model(inp, x, name="LSTM")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_patchtst(n_patches: int, patch_len: int, n_features: int,
                    horizon: int, d_model: int, n_heads: int,
                    n_layers: int, ffn_dim: int, dropout: float) -> keras.Model:
    """
    PatchTST: patch embedding → Transformer encoder → linear projection.
    Each patch is linearly projected to d_model, then positional encoding
    is added, and n_layers of multi-head self-attention are applied.
    """
    # Input: (batch, n_patches, patch_len, n_features)
    inp = keras.Input(shape=(n_patches, patch_len, n_features))

    # Flatten each patch: (batch, n_patches, patch_len * n_features)
    x = layers.Reshape((n_patches, patch_len * n_features))(inp)

    # Patch embedding
    x = layers.Dense(d_model)(x)            # (batch, n_patches, d_model)

    # Learned positional encoding
    pos_enc = layers.Embedding(input_dim=n_patches, output_dim=d_model)
    positions = tf.range(start=0, limit=n_patches, delta=1)
    x = x + pos_enc(positions)              # broadcast over batch

    # Transformer encoder layers
    for _ in range(n_layers):
        # Multi-head self-attention
        attn_out = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        # Feed-forward
        ff = layers.Dense(ffn_dim, activation="relu")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(d_model)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    # Global average pooling over patches
    x = layers.GlobalAveragePooling1D()(x)   # (batch, d_model)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)           # (batch, horizon)

    model = keras.Model(inp, out, name="PatchTST")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss="mse", metrics=["mae"])
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def run(sample_id: int = SAMPLE_ID,
        output_dir: Path = OUTPUT_DIR,
        forecast_steps: int = FORECAST_STEPS,
        test_frac: float = TEST_FRAC) -> dict:

    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    ts_path = output_dir / "ts_df.parquet"
    if not ts_path.exists():
        raise FileNotFoundError(
            f"{ts_path} not found. Run 01_data_pipeline.py first.")

    ts_df = pd.read_parquet(ts_path)
    ids = ts_df["norad_id"].unique().tolist()
    if sample_id not in ids:
        sample_id = int(ids[0])
        log.warning(f"NORAD {SAMPLE_ID} not found — using {sample_id} instead")

    sat = (ts_df[ts_df["norad_id"] == sample_id]
           .sort_values("timestamp")
           .reset_index(drop=True))

    # Feature matrix: x, y, z (km) — differenced once for stationarity
    xyz      = sat[["x", "y", "z"]].values.astype(np.float64)
    xyz_diff = np.diff(xyz, axis=0)        # (T-1, 3)
    T        = len(xyz_diff)

    n_test  = min(forecast_steps, int(T * test_frac))
    n_train = T - n_test

    train_raw = xyz_diff[:n_train]
    test_raw  = xyz_diff[n_train:]

    # Scale
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler_X.fit_transform(train_raw)
    test_sc  = scaler_X.transform(test_raw)
    full_sc  = np.vstack([train_sc, test_sc])

    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(train_raw[:, :1])

    metrics = {}

    # ── 1. VAR ────────────────────────────────────────────────────────────────
    log.info("Training VAR …")
    var_result = VAR(train_raw).fit(maxlags=VAR_MAX_LAGS, ic="aic")
    best_lag = var_result.k_ar
    log.info(f"  Best VAR lag = {best_lag}")

    # In-sample RMSE on test
    var_forecast = var_result.forecast(train_raw[-best_lag:], steps=n_test)
    var_abs_pred = xyz[n_train:n_train + n_test] + np.cumsum(var_forecast, axis=0)
    var_abs_true = xyz[n_train + 1 : n_train + n_test + 1]
    min_len = min(len(var_abs_pred), len(var_abs_true))
    var_rmse = float(np.sqrt(np.mean(
        (var_abs_pred[:min_len, 0] - var_abs_true[:min_len, 0]) ** 2)))
    metrics["VAR"] = {"lag": best_lag, "RMSE_km": round(var_rmse, 2)}
    log.info(f"  VAR RMSE = {var_rmse:.1f} km")

    var_pkl = models_dir / "var_model.pkl"
    with open(var_pkl, "wb") as f:
        pickle.dump({"result": var_result,
                     "best_lag": best_lag,
                     "train_tail": train_raw[-best_lag:],
                     "last_abs": xyz[n_train]}, f)
    log.info(f"  Saved → {var_pkl}")

    # ── 2. LSTM ───────────────────────────────────────────────────────────────
    log.info("Training LSTM …")
    X_tr, y_tr = _make_sequences(train_sc, LSTM_LOOKBACK, 1)
    X_te, y_te = _make_sequences(full_sc,  LSTM_LOOKBACK, 1)
    # Only keep test portion
    X_te = X_te[n_train - LSTM_LOOKBACK :]
    y_te = y_te[n_train - LSTM_LOOKBACK :]

    lstm_model = _build_lstm(LSTM_LOOKBACK, 3, 1, LSTM_UNITS, LSTM_DROPOUT)
    cb = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                        monitor="val_loss"),
          keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    hist = lstm_model.fit(X_tr, y_tr, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                          validation_split=0.1, callbacks=cb, verbose=0)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    val_mse    = float(np.min(hist.history["val_loss"]))
    log.info(f"  LSTM best epoch = {best_epoch}, val MSE = {val_mse:.6f}")

    lstm_path = models_dir / "lstm_model.keras"
    lstm_model.save(lstm_path)
    log.info(f"  Saved → {lstm_path}")
    metrics["LSTM"] = {"best_epoch": best_epoch, "val_MSE": round(val_mse, 6)}

    # ── 3. PatchTST ───────────────────────────────────────────────────────────
    log.info("Training PatchTST …")
    X_ptr, y_ptr = _make_patch_sequences(train_sc, PATCH_LEN, PATCH_STRIDE, 1)
    X_pte, y_pte = _make_patch_sequences(full_sc,  PATCH_LEN, PATCH_STRIDE, 1)
    n_patches = X_ptr.shape[1]
    patch_lookback = PATCH_LEN * 4

    # Only test portion
    offset = n_train - patch_lookback
    X_pte = X_pte[max(0, offset):]
    y_pte = y_pte[max(0, offset):]

    ptst_model = _build_patchtst(
        n_patches, PATCH_LEN, 3, 1,
        D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, PTST_DROPOUT)

    log.info(f"  PatchTST params: {ptst_model.count_params():,}  "
             f"| n_patches={n_patches}")

    cb_pt = [keras.callbacks.EarlyStopping(patience=5,
                                            restore_best_weights=True,
                                            monitor="val_loss"),
             keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    hist_pt = ptst_model.fit(X_ptr, y_ptr,
                             epochs=PTST_EPOCHS, batch_size=PTST_BATCH,
                             validation_split=0.1,
                             callbacks=cb_pt, verbose=0)
    best_ep_pt = int(np.argmin(hist_pt.history["val_loss"])) + 1
    val_mse_pt = float(np.min(hist_pt.history["val_loss"]))
    log.info(f"  PatchTST best epoch = {best_ep_pt}, val MSE = {val_mse_pt:.6f}")

    ptst_path = models_dir / "patchtst_model.keras"
    ptst_model.save(ptst_path)
    log.info(f"  Saved → {ptst_path}")
    metrics["PatchTST"] = {"best_epoch": best_ep_pt,
                           "val_MSE": round(val_mse_pt, 6),
                           "n_patches": n_patches,
                           "patch_len": PATCH_LEN}

    # ── Save scalers & config ─────────────────────────────────────────────────
    with open(models_dir / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(models_dir / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    config = {
        "run_timestamp":  datetime.now(timezone.utc).isoformat(),
        "sample_id":      sample_id,
        "n_train":        n_train,
        "n_test":         n_test,
        "forecast_steps": forecast_steps,
        "var_max_lags":   VAR_MAX_LAGS,
        "lstm_lookback":  LSTM_LOOKBACK,
        "lstm_units":     LSTM_UNITS,
        "lstm_epochs":    LSTM_EPOCHS,
        "patch_len":      PATCH_LEN,
        "patch_stride":   PATCH_STRIDE,
        "d_model":        D_MODEL,
        "n_heads":        N_HEADS,
        "n_layers":       N_LAYERS,
        "ptst_epochs":    PTST_EPOCHS,
        "metrics":        metrics,
    }
    with open(models_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("✅ Training complete. Models saved to %s", models_dir)
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space Debris — Model Training")
    parser.add_argument("--sample-id",   type=int, default=SAMPLE_ID)
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--forecast-steps", type=int, default=FORECAST_STEPS)
    args = parser.parse_args()

    run(sample_id=args.sample_id,
        output_dir=Path(args.output_dir),
        forecast_steps=args.forecast_steps)
