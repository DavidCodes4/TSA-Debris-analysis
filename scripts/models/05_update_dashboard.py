"""
05_update_dashboard.py
========================
Consolidate all pipeline outputs into a single dashboard-ready JSON
and update the Flask API's local data files so the React dashboard
immediately reflects the latest forecasts and conjunction alerts.

What it does
------------
1. Reads  Output/forecast_results.json         (from 03_evaluate_models.py)
2. Reads  Output/conjunction_report.json       (from 04_conjunction_assessment.py)
3. Reads  Output/benchmark_table.csv           (from 03_evaluate_models.py)
4. Reads  Output/metadata.json                 (from 01_data_pipeline.py)
5. Writes Output/dashboard_state.json          — single source of truth consumed
                                                 by GET /api/dashboard/stats and
                                                 a new GET /api/tsa/state endpoint
6. Writes Output/space_debris_catalog.csv      — debris catalog for /api/debris
7. Prints a human-readable summary to stdout

The dashboard_api.py already uses Output/ as its LOCAL fallback.
After this script finishes you can start (or reload) dashboard_api.py
and the UI will show the latest TSA results.
"""

import json, logging, argparse, math, shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Output"


def _safe(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    except (TypeError, ValueError):
        return v


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir = Path(output_dir)

    # ── Load all pipeline outputs ─────────────────────────────────────────────
    meta       = _load_json(output_dir / "metadata.json")
    forecast   = _load_json(output_dir / "forecast_results.json")
    conj       = _load_json(output_dir / "conjunction_report.json")
    bench_path = output_dir / "benchmark_table.csv"
    bench_df   = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()

    # ── Derive dashboard stats ─────────────────────────────────────────────────
    alerts = conj.get("alerts", [])
    alerts_df = pd.DataFrame(alerts) if alerts else pd.DataFrame()

    risk_counts = {}
    if not alerts_df.empty and "risk_level" in alerts_df.columns:
        risk_counts = alerts_df["risk_level"].value_counts().to_dict()

    # Closest approach
    closest = None
    if not alerts_df.empty and "distance_km" in alerts_df.columns:
        row = alerts_df.loc[alerts_df["distance_km"].idxmin()]
        closest = {
            "norad_id_1":   str(row.get("norad_id_1", "?")),
            "norad_id_2":   str(row.get("norad_id_2", "?")),
            "distance_km":  _safe(row.get("distance_km")),
            "risk_level":   str(row.get("risk_level", "?")),
            "epoch_1":      str(row.get("detection_timestamp", "")),
        }

    best_model  = forecast.get("best_model", "unknown")
    benchmark   = forecast.get("benchmark", {})

    # ML metrics block (for /api/ml/metrics)
    ml_metrics = {
        "best_model":   best_model,
        "models":       benchmark,
        "run_timestamp": forecast.get("run_timestamp", ""),
    }

    # ── Dashboard state ───────────────────────────────────────────────────────
    state = {
        "generated_at":             datetime.now(timezone.utc).isoformat(),
        "data_pipeline": {
            "tle_file":     meta.get("tle_file", ""),
            "n_objects":    meta.get("n_objects", 0),
            "n_timesteps":  meta.get("n_timesteps", 0),
            "resample_min": meta.get("resample_min", 5),
        },
        "forecast": {
            "sample_id":      forecast.get("sample_id"),
            "forecast_steps": forecast.get("forecast_steps"),
            "best_model":     best_model,
            "benchmark":      benchmark,
        },
        "conjunction": {
            "n_alerts":     conj.get("n_alerts", 0),
            "risk_summary": conj.get("risk_summary", {}),
            "screen_km":    conj.get("screen_km", 50),
            "screen_hours": conj.get("screen_hours", 24),
        },
        "stats": {
            "total_debris_objects":     meta.get("n_objects", 0),
            "active_satellites":        meta.get("n_objects", 0),
            "critical_risk_collisions": risk_counts.get("CRITICAL", 0),
            "high_risk_collisions":     risk_counts.get("HIGH", 0),
            "medium_risk_collisions":   risk_counts.get("MEDIUM", 0),
            "low_risk_collisions":      risk_counts.get("LOW", 0),
            "total_active_collisions":  len(alerts_df),
            "total_collision_pairs":    len(alerts_df),
            "min_distance_km":          _safe(alerts_df["distance_km"].min())
                                        if not alerts_df.empty else None,
            "avg_distance_km":          _safe(alerts_df["distance_km"].mean())
                                        if not alerts_df.empty else None,
            "max_distance_km":          _safe(alerts_df["distance_km"].max())
                                        if not alerts_df.empty else None,
            "closest_approach":         closest,
            "simulated_time":           datetime.now(timezone.utc).isoformat(),
            "timestamp":                datetime.now(timezone.utc).isoformat(),
        },
        "ml_metrics": ml_metrics,
    }

    # ── Save dashboard_state.json ─────────────────────────────────────────────
    state_path = output_dir / "dashboard_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.info(f"Saved → {state_path}")

    # ── Build space_debris_catalog.csv (for /api/debris) ─────────────────────
    ts_path = output_dir / "ts_df.parquet"
    if ts_path.exists():
        ts_df = pd.read_parquet(ts_path)
        catalog = (ts_df.groupby("norad_id").agg(
            OBJECT_NAME =("norad_id", lambda x: str(x.iloc[0])),
            ALTITUDE_KM =("altitude_km", "mean"),
            SPEED_KMS   =("speed_kms",   "mean"),
            N_STEPS     =("timestamp",   "count"),
        ).reset_index().rename(columns={"norad_id": "NORAD_CAT_ID"}))
        catalog["COUNTRY"]     = "UNKNOWN"
        catalog["LAUNCH"]      = ""
        catalog["OBJECT_TYPE"] = "DEBRIS"
        catalog["RCS_SIZE"]    = "UNKNOWN"
        catalog["INCLINATION"] = None
        catalog["APOGEE"]      = catalog["ALTITUDE_KM"]
        catalog["PERIGEE"]     = catalog["ALTITUDE_KM"]
        catalog["PERIOD"]      = None

        cat_path = output_dir / "space_debris_catalog.csv"
        catalog.to_csv(cat_path, index=False)
        log.info(f"Saved → {cat_path}  ({len(catalog)} objects)")

    # ── Print human-readable summary ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SPACE DEBRIS TSA PIPELINE — DASHBOARD UPDATE SUMMARY")
    print("=" * 65)
    print(f"  Generated    : {state['generated_at']}")
    print(f"  TLE file     : {state['data_pipeline']['tle_file']}")
    print(f"  Objects      : {state['data_pipeline']['n_objects']}")
    print(f"  Timesteps    : {state['data_pipeline']['n_timesteps']:,}")
    print(f"  Best model   : {best_model}")
    if benchmark:
        best = benchmark.get(best_model, {})
        print(f"  Best RMSE    : {best.get('RMSE', '?')} km")
        print(f"  Best MAE     : {best.get('MAE', '?')} km")
    print(f"  Alerts total : {conj.get('n_alerts', 0)}")
    for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        cnt = risk_counts.get(lvl, 0)
        if cnt:
            print(f"    {lvl:<10}: {cnt}")
    if closest:
        print(f"  Closest pair : NORAD {closest['norad_id_1']} ↔ "
              f"{closest['norad_id_2']}  "
              f"({closest['distance_km']} km, {closest['risk_level']})")
    print("=" * 65)
    print(f"\n  ✅ Dashboard ready — start / reload dashboard_api.py\n")

    return state


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Space Debris — Dashboard Update")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(output_dir=Path(args.output_dir))
