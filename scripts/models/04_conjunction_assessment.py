"""
04_conjunction_assessment.py
==============================
Multi-object conjunction (close-approach) screening.

For every pair of objects in ts_df, find the time of closest approach (TCA)
within the forecast window and assign a risk level.

Risk thresholds (standard NASA/ESA screening):
  CRITICAL : distance < 1 km
  HIGH     : 1  ≤ distance < 5  km
  MEDIUM   : 5  ≤ distance < 20 km
  LOW      : 20 ≤ distance < 50 km
  (pairs > 50 km are not reported)

Outputs (OUTPUT_DIR):
  collision_alerts_forecast/part-00000.csv  ← dashboard_api.py CSV fallback
  conjunction_report.json                   ← full machine-readable report
"""

import json, logging, argparse, math
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR     = Path(__file__).resolve().parents[2] / "Output"

# Screening thresholds (km)
SCREEN_KM      = 50.0
CRITICAL_KM    = 1.0
HIGH_KM        = 5.0
MEDIUM_KM      = 20.0

# Screen only the first SCREEN_HOURS hours of the common time window
SCREEN_HOURS   = 24.0
MAX_PAIRS      = 5000     # cap to avoid O(n²) explosion


def _risk(dist_km: float) -> str:
    if dist_km < CRITICAL_KM:
        return "CRITICAL"
    if dist_km < HIGH_KM:
        return "HIGH"
    if dist_km < MEDIUM_KM:
        return "MEDIUM"
    return "LOW"


def _collision_probability(dist_km: float,
                            rel_vel_kms: float,
                            combined_radius_km: float = 0.01) -> float:
    """
    Simple Gaussian miss-distance probability (Chan 2008 approximation).
    P ≈ (π r²) / (2π σ²) where σ = max(dist_km/3, combined_radius_km)
    """
    sigma = max(dist_km / 3.0, combined_radius_km)
    r     = combined_radius_km
    p     = (math.pi * r ** 2) / (2 * math.pi * sigma ** 2)
    return round(min(p, 1.0), 8)


def run(output_dir: Path = OUTPUT_DIR,
        screen_km: float = SCREEN_KM,
        screen_hours: float = SCREEN_HOURS) -> pd.DataFrame:

    output_dir = Path(output_dir)
    ts_path    = output_dir / "ts_df.parquet"

    if not ts_path.exists():
        raise FileNotFoundError(
            f"{ts_path} not found. Run 01_data_pipeline.py first.")

    ts_df = pd.read_parquet(ts_path)
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], utc=True)

    # Pivot to wide format: one row per timestamp, columns = (norad_id, axis)
    ids = ts_df["norad_id"].unique().tolist()
    log.info(f"Screening {len(ids)} objects for conjunctions …")

    # Find common time window
    t_start = ts_df.groupby("norad_id")["timestamp"].min().max()
    t_end   = t_start + pd.Timedelta(hours=screen_hours)
    window  = ts_df[(ts_df["timestamp"] >= t_start) &
                    (ts_df["timestamp"] <= t_end)].copy()

    # Build position dict: norad_id → (T, 3) array indexed by common timestamps
    common_ts = sorted(window["timestamp"].unique())
    pos = {}
    for nid in ids:
        sub = (window[window["norad_id"] == nid]
               .set_index("timestamp")[["x", "y", "z"]])
        sub = sub.reindex(common_ts)
        if sub.isnull().any().any():
            sub = sub.interpolate(method="time", limit_direction="both")
        if not sub.isnull().any().any():
            pos[nid] = sub.values   # (T, 3)

    valid_ids = list(pos.keys())
    log.info(f"  Objects with full coverage: {len(valid_ids)}")

    if len(valid_ids) < 2:
        log.warning("Not enough objects for conjunction screening.")
        return pd.DataFrame()

    # Limit pairs
    pair_list = list(combinations(valid_ids, 2))
    if len(pair_list) > MAX_PAIRS:
        log.warning(f"Capping to {MAX_PAIRS} random pairs (of {len(pair_list)})")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pair_list), MAX_PAIRS, replace=False)
        pair_list = [pair_list[i] for i in idx]

    alerts = []
    for n1, n2 in pair_list:
        p1 = pos[n1]   # (T, 3)
        p2 = pos[n2]   # (T, 3)
        diffs = p1 - p2
        dists = np.linalg.norm(diffs, axis=1)   # (T,)
        tca_idx = int(np.argmin(dists))
        tca_dist = float(dists[tca_idx])

        if tca_dist > screen_km:
            continue

        tca_ts  = common_ts[tca_idx]
        risk    = _risk(tca_dist)

        # Relative velocity at TCA
        if tca_idx > 0:
            dt_sec  = (common_ts[tca_idx] -
                       common_ts[tca_idx - 1]).total_seconds()
            dv1 = (p1[tca_idx] - p1[tca_idx - 1]) / dt_sec   # km/s
            dv2 = (p2[tca_idx] - p2[tca_idx - 1]) / dt_sec
            rel_vel = float(np.linalg.norm(dv1 - dv2))
        else:
            rel_vel = 0.0

        # Altitude at TCA
        R_EARTH = 6371.0
        alt1 = float(np.linalg.norm(p1[tca_idx])) - R_EARTH
        alt2 = float(np.linalg.norm(p2[tca_idx])) - R_EARTH

        prob = _collision_probability(tca_dist, rel_vel)

        alerts.append({
            "norad_id_1":             n1,
            "norad_id_2":             n2,
            "object_name_1":          str(n1),
            "object_name_2":          str(n2),
            "type_1":                 "UNKNOWN",
            "type_2":                 "UNKNOWN",
            "distance_km":            round(tca_dist, 4),
            "risk_level":             risk,
            "collision_probability":  prob,
            "relative_velocity_kms":  round(rel_vel, 4),
            "altitude_km_1":          round(alt1, 2),
            "altitude_km_2":          round(alt2, 2),
            "pos_x_1":                round(float(p1[tca_idx, 0]), 4),
            "pos_y_1":                round(float(p1[tca_idx, 1]), 4),
            "pos_z_1":                round(float(p1[tca_idx, 2]), 4),
            "pos_x_2":                round(float(p2[tca_idx, 0]), 4),
            "pos_y_2":                round(float(p2[tca_idx, 1]), 4),
            "pos_z_2":                round(float(p2[tca_idx, 2]), 4),
            "detection_timestamp":    pd.Timestamp(tca_ts).isoformat(),
            "collision_type":         f"TSA-predicted ({risk})",
        })

    alerts_df = pd.DataFrame(alerts)

    if alerts_df.empty:
        log.info("No conjunctions found within screening distance.")
    else:
        alerts_df = alerts_df.sort_values("distance_km").reset_index(drop=True)
        risk_counts = alerts_df["risk_level"].value_counts().to_dict()
        log.info(f"Conjunctions found: {len(alerts_df)}  "
                 f"| CRITICAL={risk_counts.get('CRITICAL',0)}"
                 f"  HIGH={risk_counts.get('HIGH',0)}"
                 f"  MEDIUM={risk_counts.get('MEDIUM',0)}"
                 f"  LOW={risk_counts.get('LOW',0)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    # 1. Dashboard-compatible CSV (matches dashboard_api.py LOCAL_COLLISION_GLOB)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = output_dir / f"collision_alerts_{ts_tag}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "part-00000.csv"
    alerts_df.to_csv(csv_path, index=False)
    log.info(f"Saved → {csv_path}")

    # 2. JSON report
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_objects":     len(valid_ids),
        "n_pairs":       len(pair_list),
        "screen_km":     screen_km,
        "screen_hours":  screen_hours,
        "n_alerts":      len(alerts_df),
        "risk_summary":  alerts_df["risk_level"].value_counts().to_dict()
                         if not alerts_df.empty else {},
        "alerts":        alerts_df.to_dict(orient="records"),
    }
    with open(output_dir / "conjunction_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Saved → {output_dir / 'conjunction_report.json'}")

    log.info("✅ Conjunction assessment complete.")
    return alerts_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Space Debris — Conjunction Assessment")
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--screen-km",     type=float, default=SCREEN_KM)
    parser.add_argument("--screen-hours",  type=float, default=SCREEN_HOURS)
    args = parser.parse_args()

    run(output_dir=Path(args.output_dir),
        screen_km=args.screen_km,
        screen_hours=args.screen_hours)
