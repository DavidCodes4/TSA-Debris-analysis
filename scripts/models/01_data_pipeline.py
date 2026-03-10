"""
01_data_pipeline.py
====================
TLE → SGP4 densification → clean resampled time-series.

DIFFERENCE vs notebook:
  Instead of linear-interpolating between raw TLE epochs (which created
  piecewise-linear artifacts), this script propagates each TLE forward
  at RESAMPLE_MIN-minute steps using sgp4, giving physically accurate
  sinusoidal orbits.

Outputs (saved to OUTPUT_DIR):
  ts_df.parquet          — resampled ECI positions for all objects
  metadata.json          — run stats, object list, config
"""

import os, json, logging, argparse, glob
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR    = Path(__file__).resolve().parents[2] / "Output"
TLE_FILE      = "tle2024.txt"          # change to process a different year
MAX_RECORDS   = 150_000                # TLE lines to read  (≈50k objects)
TOP_N         = 200                    # objects to keep after filtering
RESAMPLE_MIN  = 5                      # minutes between propagated steps
PROPAGATE_HRS = 24 * 3                 # hours forward to propagate each TLE
MIN_SPAN_HRS  = 12                     # min hours span to keep object

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_tle_file(path: Path, max_records: int) -> pd.DataFrame:
    """Read a TLE text file and return a DataFrame of parsed fields."""
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [l.rstrip() for l in fh if l.strip()]

    i = 0
    while i + 2 < len(lines) and len(records) * 3 < max_records:
        name  = lines[i].strip()
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                norad = int(line1[2:7])
                # epoch: year + day-of-year fraction (columns 18-32)
                ep_yr = int(line1[18:20])
                ep_day = float(line1[20:32])
                year = 2000 + ep_yr if ep_yr < 57 else 1900 + ep_yr
                epoch_dt = (pd.Timestamp(year=year, month=1, day=1, tz="UTC")
                            + pd.Timedelta(days=ep_day - 1))
                records.append({
                    "norad_id": norad,
                    "name":     name,
                    "tle1":     line1,
                    "tle2":     line2,
                    "epoch":    epoch_dt,
                })
            except Exception:
                pass
        i += 3

    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df):,} TLE records from {path.name}")
    return df


def _densify_object(norad_id: int, tle1: str, tle2: str,
                    epoch: pd.Timestamp,
                    resample_min: int,
                    propagate_hrs: float) -> pd.DataFrame:
    """
    SGP4-propagate one TLE forward at resample_min-minute steps.
    Returns DataFrame with columns: norad_id, timestamp, x, y, z (km).
    """
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
    except Exception:
        return pd.DataFrame()

    rows = []
    n_steps = int(propagate_hrs * 60 / resample_min)
    for step in range(n_steps + 1):
        ts = epoch + pd.Timedelta(minutes=step * resample_min)
        jd, fr = jday(ts.year, ts.month, ts.day,
                      ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:                           # e==0 means success
            rows.append({
                "norad_id":  norad_id,
                "timestamp": ts,
                "x":         r[0],
                "y":         r[1],
                "z":         r[2],
                "vx":        v[0],
                "vy":        v[1],
                "vz":        v[2],
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(tle_file: str = TLE_FILE,
        max_records: int = MAX_RECORDS,
        top_n: int = TOP_N,
        resample_min: int = RESAMPLE_MIN,
        propagate_hrs: float = PROPAGATE_HRS,
        min_span_hrs: float = MIN_SPAN_HRS,
        output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse TLE file(s)
    # Special value "ALL" → load every .txt file in RAW_DIR
    if tle_file.upper() == "ALL":
        tle_paths = sorted(RAW_DIR.glob("*.txt"))
        if not tle_paths:
            raise FileNotFoundError(f"No TLE .txt files found in {RAW_DIR}")
        log.info(f"Loading ALL {len(tle_paths)} TLE files from {RAW_DIR} …")
        frames = []
        per_file = max(max_records // len(tle_paths), 5000)
        for p in tle_paths:
            try:
                frames.append(_parse_tle_file(p, per_file))
            except Exception as e:
                log.warning(f"  Skipping {p.name}: {e}")
        raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        used_tle_label = "ALL"
    else:
        tle_path = RAW_DIR / tle_file
        raw_df = _parse_tle_file(tle_path, max_records)
        used_tle_label = tle_file

    # 2. Keep one TLE per object (latest epoch across all files)
    raw_df = (raw_df.sort_values("epoch")
              .groupby("norad_id", as_index=False).last())
    log.info(f"Unique objects across all sources: {len(raw_df):,}")

    # 3. Select TOP_N by recency (largest epoch)
    candidates = raw_df.nlargest(top_n * 3, "epoch")   # over-select then filter

    # 4. SGP4 densification — ALL objects propagated from the SAME reference
    #    start time so their position grids overlap and conjunction screening works.
    #    Use the median epoch of the selected candidates as the common start,
    #    which keeps propagation error small while ensuring overlap.
    ref_start = pd.Timestamp(candidates["epoch"].median()).tz_localize("UTC") \
        if candidates["epoch"].median().tzinfo is None \
        else pd.Timestamp(candidates["epoch"].median()).tz_convert("UTC")
    log.info(f"SGP4 densification ({resample_min}-min steps, "
             f"{propagate_hrs}h horizon, ref_start={ref_start.date()}) …")
    all_ts = []
    ok = 0
    for _, row in candidates.iterrows():
        ts_df_obj = _densify_object(
            row["norad_id"], row["tle1"], row["tle2"],
            ref_start,          # ← common start for all objects
            resample_min, propagate_hrs)
        if len(ts_df_obj) < 2:
            continue
        span_hrs = (ts_df_obj["timestamp"].max()
                    - ts_df_obj["timestamp"].min()).total_seconds() / 3600
        if span_hrs < min_span_hrs:
            continue
        all_ts.append(ts_df_obj)
        ok += 1
        if ok >= top_n:
            break

    if not all_ts:
        raise RuntimeError("No objects survived the span filter. "
                           "Lower min_span_hrs or increase propagate_hrs.")

    ts_df = pd.concat(all_ts, ignore_index=True)
    ts_df = ts_df.sort_values(["norad_id", "timestamp"]).reset_index(drop=True)

    # 5. Altitude (from ECI position magnitude)
    R_EARTH = 6371.0
    ts_df["altitude_km"] = np.sqrt(ts_df["x"]**2 + ts_df["y"]**2
                                   + ts_df["z"]**2) - R_EARTH
    ts_df["speed_kms"] = np.sqrt(ts_df["vx"]**2 + ts_df["vy"]**2
                                 + ts_df["vz"]**2)

    n_objects = ts_df["norad_id"].nunique()
    n_steps   = len(ts_df)
    log.info(f"✅ Densified: {n_objects} objects, {n_steps:,} total timesteps")

    # 6. Save
    out_path = output_dir / "ts_df.parquet"
    ts_df.to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path}")

    meta = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "tle_file":      used_tle_label,
        "ref_start":     ref_start.isoformat(),
        "n_objects":     n_objects,
        "n_timesteps":   n_steps,
        "resample_min":  resample_min,
        "propagate_hrs": propagate_hrs,
        "min_span_hrs":  min_span_hrs,
        "object_ids":    ts_df["norad_id"].unique().tolist(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Data pipeline complete.")
    return ts_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space Debris — Data Pipeline")
    parser.add_argument("--tle-file",       default=TLE_FILE,
                        help="TLE filename in data/raw/, or 'ALL' to load every year")
    parser.add_argument("--max-records",    type=int, default=MAX_RECORDS)
    parser.add_argument("--top-n",          type=int, default=TOP_N)
    parser.add_argument("--resample-min",   type=int, default=RESAMPLE_MIN)
    parser.add_argument("--propagate-hrs",  type=float, default=PROPAGATE_HRS)
    parser.add_argument("--min-span-hrs",   type=float, default=MIN_SPAN_HRS)
    parser.add_argument("--output-dir",     default=str(OUTPUT_DIR))
    args = parser.parse_args()

    run(tle_file=args.tle_file,
        max_records=args.max_records,
        top_n=args.top_n,
        resample_min=args.resample_min,
        propagate_hrs=args.propagate_hrs,
        min_span_hrs=args.min_span_hrs,
        output_dir=Path(args.output_dir))
