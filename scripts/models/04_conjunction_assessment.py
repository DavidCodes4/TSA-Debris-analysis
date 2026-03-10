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
  collision_history.parquet                 ← cumulative history across all runs
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
SCREEN_KM      = 100.0     # widened: catches more real close-approaches
CRITICAL_KM    = 1.0
HIGH_KM        = 5.0
MEDIUM_KM      = 20.0

# Screen only the first SCREEN_HOURS hours of the common time window
SCREEN_HOURS   = 72.0      # full propagation window
MAX_PAIRS      = 19900     # screen ALL pairs for 200 objects (200*199/2)


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
        screen_hours: float = SCREEN_HOURS,
        tle_file: str = "unknown",
        append_history: bool = True) -> pd.DataFrame:

    output_dir = Path(output_dir)
    ts_path    = output_dir / "ts_df.parquet"

    if not ts_path.exists():
        raise FileNotFoundError(
            f"{ts_path} not found. Run 01_data_pipeline.py first.")

    ts_df = pd.read_parquet(ts_path)
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], utc=True)

    ids = ts_df["norad_id"].unique().tolist()
    log.info(f"Screening {len(ids)} objects for conjunctions …")

    # ── Build per-object position series (no global grid requirement) ─────────
    # Each object keeps its own timestamps; we interpolate only within pairs.
    RESAMPLE = "5min"   # normalise to a fixed 5-min grid per object
    pos = {}            # norad_id → DataFrame(index=timestamp, cols=[x,y,z])
    for nid in ids:
        sub = (ts_df[ts_df["norad_id"] == nid]
               .set_index("timestamp")[["x", "y", "z"]]
               .sort_index())
        # Clip to screen_hours window starting from this object's first timestamp
        t0 = sub.index.min()
        sub = sub[sub.index <= t0 + pd.Timedelta(hours=screen_hours)]
        if len(sub) < 2:
            continue
        # Resample to uniform 5-min grid using linear interpolation
        new_idx = pd.date_range(sub.index.min(), sub.index.max(), freq=RESAMPLE)
        sub = sub.reindex(sub.index.union(new_idx)).interpolate("time").reindex(new_idx)
        if sub.isnull().any().any():
            sub = sub.dropna()
        if len(sub) >= 2:
            pos[nid] = sub

    valid_ids = list(pos.keys())
    log.info(f"  Objects available for pairing: {len(valid_ids)}")

    if len(valid_ids) < 2:
        log.warning("Not enough objects for conjunction screening. "
                    "Try re-running step 1 (make step1) first.")
        return pd.DataFrame()

    # ── Sample pairs ─────────────────────────────────────────────────────────
    pair_list = list(combinations(valid_ids, 2))
    if len(pair_list) > MAX_PAIRS:
        log.warning(f"Capping to {MAX_PAIRS} random pairs (of {len(pair_list)})")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pair_list), MAX_PAIRS, replace=False)
        pair_list = [pair_list[i] for i in idx]

    alerts = []
    for n1, n2 in pair_list:
        df1 = pos[n1]
        df2 = pos[n2]

        # Intersect their individual time grids
        common_ts = df1.index.intersection(df2.index)
        if len(common_ts) < 2:
            # No overlap at all — interpolate n2 onto n1's grid
            common_ts = df1.index
            df2_aligned = df2.reindex(df2.index.union(common_ts)).interpolate("time").reindex(common_ts)
            if df2_aligned.isnull().any().any():
                continue
            p1 = df1.loc[common_ts].values
            p2 = df2_aligned.values
        else:
            p1 = df1.loc[common_ts].values
            p2 = df2.loc[common_ts].values

        diffs    = p1 - p2
        dists    = np.linalg.norm(diffs, axis=1)
        tca_idx  = int(np.argmin(dists))
        tca_dist = float(dists[tca_idx])

        if tca_dist > screen_km:
            continue

        tca_ts = common_ts[tca_idx]
        risk   = _risk(tca_dist)

        # Relative velocity at TCA
        if tca_idx > 0:
            dt_sec = (common_ts[tca_idx] - common_ts[tca_idx - 1]).total_seconds()
            if dt_sec > 0:
                dv1 = (p1[tca_idx] - p1[tca_idx - 1]) / dt_sec
                dv2 = (p2[tca_idx] - p2[tca_idx - 1]) / dt_sec
                rel_vel = float(np.linalg.norm(dv1 - dv2))
            else:
                rel_vel = 0.0
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
        "tle_file":      tle_file,
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

    # 3. Cumulative history parquet — append new alerts across all runs
    if append_history and not alerts_df.empty:
        history_path = output_dir / "collision_history.parquet"
        alerts_df["tle_source"] = tle_file
        alerts_df["run_tag"]    = ts_tag
        if history_path.exists():
            existing = pd.read_parquet(history_path)
            # Deduplicate by (norad_id_1, norad_id_2, detection_timestamp)
            combined = pd.concat([existing, alerts_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["norad_id_1", "norad_id_2", "detection_timestamp"],
                keep="last"
            ).reset_index(drop=True)
        else:
            combined = alerts_df
        combined.to_parquet(history_path, index=False)
        log.info(f"History: {len(combined):,} total events → {history_path}")

    log.info("✅ Conjunction assessment complete.")
    return alerts_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Space Debris — Conjunction Assessment")
    parser.add_argument("--output-dir",       default=str(OUTPUT_DIR))
    parser.add_argument("--screen-km",        type=float, default=SCREEN_KM)
    parser.add_argument("--screen-hours",     type=float, default=SCREEN_HOURS)
    parser.add_argument("--tle-file",         default="unknown")
    parser.add_argument("--no-history",       action="store_true",
                        help="Do not append to collision_history.parquet")
    args = parser.parse_args()

    run(output_dir=Path(args.output_dir),
        screen_km=args.screen_km,
        screen_hours=args.screen_hours,
        tle_file=args.tle_file,
        append_history=not args.no_history)
