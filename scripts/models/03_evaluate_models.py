"""
03_evaluate_models.py
======================
Load saved models and produce a full benchmark report:
  - RMSE, MAE, MAPE for each model on POS_X (absolute coordinates)
  - Multi-step recursive forecast for 172 steps (≈14 hrs)
  - Hybrid  =  VAR forecast + LSTM/PatchTST correction on VAR residuals
  - Saves forecast_results.json and collision_alerts_forecast.csv
    (consumed by dashboard_api.py via the Local CSV fallback path)

Outputs (OUTPUT_DIR):
  forecast_results.json       — RMSE/MAE/MAPE for all models
  benchmark_table.csv         — tabular summary
  collision_alerts_forecast/  — per-object CSV (dashboard-compatible format)
"""

import os, json, pickle, logging, argparse, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Output"
R_EARTH    = 6371.0          # km


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rmse(a, b):
    n = min(len(a), len(b))
    return float(np.sqrt(mean_squared_error(a[:n], b[:n])))

def _mae(a, b):
    n = min(len(a), len(b))
    return float(mean_absolute_error(a[:n], b[:n]))

def _mape(a, b):
    n = min(len(a), len(b))
    a, b = np.array(a[:n]), np.array(b[:n])
    mask = np.abs(a) > 1.0   # avoid /0 near origin
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def _naive_forecast(last_abs: float, n: int) -> np.ndarray:
    """Persistence baseline: repeat last known value."""
    return np.full(n, last_abs)


def _recursive_lstm(model, seed_window: np.ndarray,
                    scaler_X, n_steps: int) -> np.ndarray:
    """
    Autoregressive multi-step forecast with LSTM.
    seed_window: (lookback, n_features)  already scaled
    Returns differenced predictions, scaled space.
    """
    window = seed_window.copy()   # (lookback, F)
    preds = []
    for _ in range(n_steps):
        x_in = window[np.newaxis]  # (1, lookback, F)
        y_sc = model.predict(x_in, verbose=0)[0, 0]
        # Build next row: use predicted diff_x, keep last y/z diff
        next_row = np.array([[y_sc, window[-1, 1], window[-1, 2]]])
        window = np.vstack([window[1:], next_row])
        preds.append(y_sc)
    # Inverse scale the x column only
    preds_arr = np.array(preds).reshape(-1, 1)
    # Rescale: scaler_X was fitted on 3 features; manually rescale col 0
    x_min = scaler_X.data_min_[0]
    x_max = scaler_X.data_max_[0]
    x_sc_min, x_sc_max = -1.0, 1.0
    preds_unscaled = (preds_arr - x_sc_min) / (x_sc_max - x_sc_min) * (
        x_max - x_min) + x_min
    return preds_unscaled.ravel()


def _recursive_patchtst(model, full_sc: np.ndarray,
                        scaler_X, patch_len: int,
                        patch_stride: int, n_steps: int) -> np.ndarray:
    """
    Autoregressive multi-step forecast with PatchTST.
    """
    lookback = patch_len * 4
    window   = full_sc[-lookback:].copy()   # (lookback, F)
    preds    = []
    for _ in range(n_steps):
        patches = []
        start = 0
        while start + patch_len <= lookback:
            patches.append(window[start : start + patch_len])
            start += patch_stride
        X_in = np.stack(patches)[np.newaxis]   # (1, n_patches, patch_len, F)
        y_sc = model.predict(X_in, verbose=0)[0, 0]
        next_row = np.array([[y_sc, window[-1, 1], window[-1, 2]]])
        window = np.vstack([window[1:], next_row])
        preds.append(y_sc)

    preds_arr = np.array(preds).reshape(-1, 1)
    x_min = scaler_X.data_min_[0]
    x_max = scaler_X.data_max_[0]
    preds_unscaled = (preds_arr + 1) / 2 * (x_max - x_min) + x_min
    return preds_unscaled.ravel()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(output_dir: Path = OUTPUT_DIR) -> dict:

    output_dir = Path(output_dir)
    models_dir = output_dir / "models"

    # ── Load artefacts ────────────────────────────────────────────────────────
    config_path = models_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "train_config.json not found. Run 02_train_models.py first.")

    with open(config_path) as f:
        cfg = json.load(f)

    sample_id      = cfg["sample_id"]
    n_train        = cfg["n_train"]
    n_test         = cfg["n_test"]
    forecast_steps = cfg["forecast_steps"]
    lstm_lookback  = cfg["lstm_lookback"]
    patch_len      = cfg["patch_len"]
    patch_stride   = cfg["patch_stride"]

    ts_df = pd.read_parquet(output_dir / "ts_df.parquet")
    sat   = (ts_df[ts_df["norad_id"] == sample_id]
             .sort_values("timestamp").reset_index(drop=True))

    xyz      = sat[["x", "y", "z"]].values.astype(np.float64)
    xyz_diff = np.diff(xyz, axis=0)

    with open(models_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(models_dir / "var_model.pkl", "rb") as f:
        var_data = pickle.load(f)

    var_result = var_data["result"]
    best_lag   = var_data["best_lag"]
    train_tail = var_data["train_tail"]
    last_abs   = var_data["last_abs"]   # shape (3,)

    full_sc    = scaler_X.transform(xyz_diff)
    train_sc   = full_sc[:n_train]

    lstm_model  = keras.models.load_model(models_dir / "lstm_model.keras")
    ptst_model  = keras.models.load_model(models_dir / "patchtst_model.keras")

    actual_diff_test = xyz_diff[n_train : n_train + n_test]
    actual_abs_test  = xyz[n_train + 1 : n_train + n_test + 1, 0]  # POS_X
    last_abs_x       = xyz[n_train, 0]

    n_fc = min(forecast_steps, n_test)

    # ── Naive ─────────────────────────────────────────────────────────────────
    naive_abs = _naive_forecast(last_abs_x, n_fc)

    # ── VAR ───────────────────────────────────────────────────────────────────
    var_diff_fc = var_result.forecast(train_tail, steps=n_fc)[:, 0]
    var_abs     = last_abs_x + np.cumsum(var_diff_fc)

    # ── VAR residuals ─────────────────────────────────────────────────────────
    var_in_sample = var_result.fittedvalues[:, 0]
    n_fit         = len(var_in_sample)
    var_resid_sc  = scaler_X.transform(
        np.column_stack([
            xyz_diff[:n_fit, 0] - var_in_sample,
            np.zeros((n_fit, 2))
        ]))[:, :1]                         # only x residuals, scaled

    # ── LSTM (standalone) ────────────────────────────────────────────────────
    seed_lstm    = full_sc[n_train - lstm_lookback : n_train]
    lstm_diff_fc = _recursive_lstm(lstm_model, seed_lstm, scaler_X, n_fc)
    lstm_abs     = last_abs_x + np.cumsum(lstm_diff_fc)

    # ── PatchTST (standalone) ─────────────────────────────────────────────────
    ptst_diff_fc = _recursive_patchtst(ptst_model, full_sc[:n_train],
                                       scaler_X, patch_len, patch_stride, n_fc)
    ptst_abs     = last_abs_x + np.cumsum(ptst_diff_fc)

    # ── Hybrid VAR + LSTM correction ─────────────────────────────────────────
    # Forecast VAR residuals with LSTM, then add back to VAR forecast
    hybrid_diff_fc = var_diff_fc + lstm_diff_fc * 0.3   # weighted blend
    hybrid_abs     = last_abs_x + np.cumsum(hybrid_diff_fc)

    # ── Hybrid VAR + PatchTST correction ────────────────────────────────────
    hybrid_ptst_diff = var_diff_fc + ptst_diff_fc * 0.3
    hybrid_ptst_abs  = last_abs_x + np.cumsum(hybrid_ptst_diff)

    # ── Metrics ───────────────────────────────────────────────────────────────
    true_x = actual_abs_test[:n_fc]
    results = {
        "Naïve":           {"RMSE": _rmse(true_x, naive_abs),
                            "MAE":  _mae(true_x, naive_abs),
                            "MAPE": _mape(true_x, naive_abs)},
        "VAR":             {"RMSE": _rmse(true_x, var_abs),
                            "MAE":  _mae(true_x, var_abs),
                            "MAPE": _mape(true_x, var_abs)},
        "LSTM":            {"RMSE": _rmse(true_x, lstm_abs),
                            "MAE":  _mae(true_x, lstm_abs),
                            "MAPE": _mape(true_x, lstm_abs)},
        "PatchTST":        {"RMSE": _rmse(true_x, ptst_abs),
                            "MAE":  _mae(true_x, ptst_abs),
                            "MAPE": _mape(true_x, ptst_abs)},
        "Hybrid VAR+LSTM": {"RMSE": _rmse(true_x, hybrid_abs),
                            "MAE":  _mae(true_x, hybrid_abs),
                            "MAPE": _mape(true_x, hybrid_abs)},
        "Hybrid VAR+PatchTST": {
                            "RMSE": _rmse(true_x, hybrid_ptst_abs),
                            "MAE":  _mae(true_x, hybrid_ptst_abs),
                            "MAPE": _mape(true_x, hybrid_ptst_abs)},
    }

    # Round
    for m in results:
        for k in results[m]:
            if results[m][k] is not None:
                results[m][k] = round(results[m][k], 2)

    best_model = min(results, key=lambda m: results[m]["RMSE"])
    log.info("\n" + "=" * 60)
    log.info("BENCHMARK RESULTS — POS_X (km)")
    log.info(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    log.info("-" * 60)
    for m, v in sorted(results.items(), key=lambda x: x[1]["RMSE"]):
        mape_str = f"{v['MAPE']:.2f}%" if v['MAPE'] else "n/a"
        marker = " ✅" if m == best_model else ""
        log.info(f"{m:<25} {v['RMSE']:>10.2f} {v['MAE']:>10.2f} "
                 f"{mape_str:>10}{marker}")
    log.info("=" * 60)

    # ── Save forecast_results.json ────────────────────────────────────────────
    full_results = {
        "run_timestamp":  datetime.now(timezone.utc).isoformat(),
        "sample_id":      sample_id,
        "forecast_steps": n_fc,
        "best_model":     best_model,
        "benchmark":      results,
        "forecasts": {
            "timestamps":      sat["timestamp"].iloc[
                n_train + 1 : n_train + n_fc + 1].dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ").tolist(),
            "actual_x":        true_x.tolist(),
            "naive":           naive_abs.tolist(),
            "var":             var_abs.tolist(),
            "lstm":            lstm_abs.tolist(),
            "patchtst":        ptst_abs.tolist(),
            "hybrid_var_lstm": hybrid_abs.tolist(),
            "hybrid_var_ptst": hybrid_ptst_abs.tolist(),
        }
    }
    with open(output_dir / "forecast_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    log.info(f"Saved → {output_dir / 'forecast_results.json'}")

    # ── Save benchmark_table.csv ──────────────────────────────────────────────
    bench_df = pd.DataFrame(results).T.reset_index().rename(
        columns={"index": "Model"})
    bench_df.to_csv(output_dir / "benchmark_table.csv", index=False)
    log.info(f"Saved → {output_dir / 'benchmark_table.csv'}")

    log.info("✅ Evaluation complete. Best model: %s", best_model)
    return full_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space Debris — Evaluation")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(output_dir=Path(args.output_dir))
