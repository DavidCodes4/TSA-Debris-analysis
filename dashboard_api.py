"""
Space Debris TSA — Dashboard REST API
======================================
All data is served from local files produced by scripts/models/run_pipeline.py.

Data sources (all under Output/):
  Output/dashboard_state.json          ← consolidated pipeline state (step 5)
  Output/forecast_results.json         ← per-model forecasts (step 3)
  Output/conjunction_report.json       ← conjunction assessment (step 4)
  Output/collision_alerts_*/part-*.csv ← raw collision alert CSVs (step 4)
  Output/space_debris_catalog.csv      ← debris catalog (step 5)

Endpoints:
  GET /api/health
  GET /api/stats
  GET /api/dashboard/stats
  GET /api/collisions               ?risk_level=&page=&per_page=&sort_by=&sort_order=
  GET /api/collisions/all
  GET /api/collisions/high-risk
  GET /api/collisions/globe
  GET /api/collisions/frequency
  GET /api/debris                   ?limit=
  GET /api/debris/<norad_id>
  GET /api/ml/metrics
  GET /api/pipeline/status
  GET /api/simulation/time
  GET /api/tsa/state
  GET /api/tsa/forecast
  GET /api/tsa/conjunctions         ?risk_level=
  GET /api/debris/trajectory        ?norad_id=&limit=&downsample=
  GET /api/debris/trajectories/all  ?limit=&downsample=&regime=

Run:
  pip install flask flask-cors pandas pyarrow
  python dashboard_api.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os, glob, json, math, logging
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not installed — pip install pandas pyarrow")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.path.dirname(__file__)) / "Output"

COLLISION_GLOB        = str(OUTPUT_DIR / "collision_alerts_*/part-*.csv")
COLLISION_GLOB_ALT    = str(OUTPUT_DIR / "collision_alerts_*.csv")
COLLISION_GLOB_STREAM = str(OUTPUT_DIR / "stream_collision_alerts_*/part-*.csv")
CATALOG_PATH          = OUTPUT_DIR / "space_debris_catalog.csv"
DASHBOARD_STATE       = OUTPUT_DIR / "dashboard_state.json"
FORECAST_RESULTS      = OUTPUT_DIR / "forecast_results.json"
CONJUNCTION_REPORT    = OUTPUT_DIR / "conjunction_report.json"
TS_DF_PARQUET         = OUTPUT_DIR / "ts_df.parquet"
TS_DF_PARQUET         = OUTPUT_DIR / "ts_df.parquet"


# ─── Column normalisation ────────────────────────────────────────────────────

def _normalise_collisions(df: "pd.DataFrame") -> "pd.DataFrame":
    col_map = {
        "NORAD_ID_1":            "norad_id_1",
        "NORAD_ID_2":            "norad_id_2",
        "DISTANCE_KM":           "distance_km",
        "RISK_LEVEL":            "risk_level",
        "COLLISION_TYPE":        "collision_type",
        "COLLISION_PROBABILITY": "collision_probability",
        "RELATIVE_VELOCITY_KMS": "relative_velocity_kms",
        "DETECTION_TIMESTAMP":   "detection_timestamp",
        "TYPE_1":                "type_1",
        "TYPE_2":                "type_2",
        "ALTITUDE_KM_1":         "altitude_km_1",
        "ALTITUDE_KM_2":         "altitude_km_2",
        "SPEED_KMS_1":           "speed_kms_1",
        "SPEED_KMS_2":           "speed_kms_2",
        "OBJECT_NAME_1":         "object_name_1",
        "OBJECT_NAME_2":         "object_name_2",
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


# ─── Data loaders ────────────────────────────────────────────────────────────

def _load_collision_data() -> "pd.DataFrame":
    """Load collision CSV files from Output/collision_alerts_* directories."""
    if not HAS_PANDAS:
        return pd.DataFrame()

    files = sorted(
        glob.glob(COLLISION_GLOB) +
        glob.glob(COLLISION_GLOB_ALT) +
        glob.glob(COLLISION_GLOB_STREAM),
        reverse=True,
    )
    if not files:
        logger.warning("No collision CSV files found in Output/")
        return pd.DataFrame()

    # Load all files from the latest batch directory
    latest_dir = str(Path(files[0]).parent)
    dfs = []
    for f in files:
        if str(Path(f).parent) == latest_dir:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                logger.debug(f"Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} collision records from {latest_dir}")
    return _normalise_collisions(df)


def _load_debris_catalog() -> "pd.DataFrame":
    """Load the debris catalog CSV produced by step 5."""
    if not HAS_PANDAS or not CATALOG_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CATALOG_PATH)
    except Exception as e:
        logger.warning(f"Could not read catalog: {e}")
        return pd.DataFrame()


def _load_dashboard_state() -> dict:
    """Load the consolidated dashboard_state.json produced by step 5."""
    if DASHBOARD_STATE.exists():
        try:
            return json.loads(DASHBOARD_STATE.read_text())
        except Exception as e:
            logger.warning(f"Could not read dashboard_state.json: {e}")
    return {}


def _load_forecast_results() -> dict:
    """Load forecast_results.json produced by step 3."""
    if FORECAST_RESULTS.exists():
        try:
            return json.loads(FORECAST_RESULTS.read_text())
        except Exception as e:
            logger.warning(f"Could not read forecast_results.json: {e}")
    return {}


def _load_conjunction_report() -> dict:
    """Load conjunction_report.json produced by step 4."""
    if CONJUNCTION_REPORT.exists():
        try:
            return json.loads(CONJUNCTION_REPORT.read_text())
        except Exception as e:
            logger.warning(f"Could not read conjunction_report.json: {e}")
    return {}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_float(v) -> "float | None":
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _no_data_yet(msg: str = "Run scripts/models/run_pipeline.py first."):
    return jsonify({"status": "no_data", "message": msg}), 404


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    state = _load_dashboard_state()
    return jsonify({
        "status":       "healthy",
        "pipeline_ran": bool(state),
        "output_dir":   str(OUTPUT_DIR),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/stats", methods=["GET"])
@app.route("/api/dashboard/stats", methods=["GET"])
def stats():
    # Primary: use pre-computed state from step 5
    state = _load_dashboard_state()
    if state.get("stats"):
        return jsonify(state["stats"])

    # Fallback: compute from raw files
    df      = _load_collision_data()
    catalog = _load_debris_catalog()

    if df.empty:
        return jsonify({
            "total_debris_objects":     len(catalog) if HAS_PANDAS and not catalog.empty else 0,
            "active_satellites":        0,
            "critical_risk_collisions": 0,
            "high_risk_collisions":     0,
            "medium_risk_collisions":   0,
            "low_risk_collisions":      0,
            "total_active_collisions":  0,
            "total_collision_pairs":    0,
            "min_distance_km":          None,
            "avg_distance_km":          None,
            "max_distance_km":          None,
            "closest_approach":         None,
            "simulated_time":           datetime.utcnow().isoformat() + "Z",
            "timestamp":                datetime.utcnow().isoformat() + "Z",
        })

    risk_counts = df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {}
    dist = df["distance_km"] if "distance_km" in df.columns else None

    closest = None
    if dist is not None:
        idx = df["distance_km"].idxmin()
        row = df.loc[idx]
        n1 = str(row.get("norad_id_1", "?"))
        n2 = str(row.get("norad_id_2", "?"))
        closest = {
            "satellite_1_id":   n1,
            "satellite_2_id":   n2,
            "satellite_1_name": str(row.get("object_name_1", n1)),
            "satellite_2_name": str(row.get("object_name_2", n2)),
            "miss_distance_km": _safe_float(row["distance_km"]),
            "risk_level":       str(row.get("risk_level", "?")),
            "predicted_time":   str(row.get("detection_timestamp", "")),
        }

    total = len(df)
    return jsonify({
        "total_debris_objects":     len(catalog) if not catalog.empty else 0,
        "active_satellites":        total,
        "critical_risk_collisions": int(risk_counts.get("CRITICAL", 0)),
        "high_risk_collisions":     int(risk_counts.get("HIGH",     0)),
        "medium_risk_collisions":   int(risk_counts.get("MEDIUM",   0)),
        "low_risk_collisions":      int(risk_counts.get("LOW",      0)),
        "total_active_collisions":  total,
        "total_collision_pairs":    total,
        "min_distance_km":          _safe_float(dist.min())  if dist is not None else None,
        "avg_distance_km":          _safe_float(dist.mean()) if dist is not None else None,
        "max_distance_km":          _safe_float(dist.max())  if dist is not None else None,
        "closest_approach":         closest,
        "simulated_time":           datetime.utcnow().isoformat() + "Z",
        "timestamp":                datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/collisions", methods=["GET"])
@app.route("/api/collisions/all", methods=["GET"])
def collisions():
    df = _load_collision_data()
    if df.empty:
        return jsonify({"count": 0, "total_count": 0, "page": 1,
                        "per_page": 50, "total_pages": 0, "collisions": []})

    # Filter
    risk_level = request.args.get("risk_level", "").upper()
    if risk_level and risk_level != "ALL" and "risk_level" in df.columns:
        df = df[df["risk_level"] == risk_level]

    # Sort
    sort_by    = request.args.get("sort_by", "distance_km")
    sort_order = request.args.get("sort_order", "asc")
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_order == "asc"))

    # Paginate
    total   = len(df)
    page    = max(1, int(request.args.get("page", 1)))
    per_pg  = min(200, max(1, int(request.args.get("per_page", 50))))
    start   = (page - 1) * per_pg
    page_df = df.iloc[start:start + per_pg]

    records = []
    for _, row in page_df.iterrows():
        n1 = str(row.get("norad_id_1", ""))
        n2 = str(row.get("norad_id_2", ""))
        records.append({
            "satellite_1_id":        n1,
            "satellite_2_id":        n2,
            "satellite_1_name":      str(row.get("object_name_1", n1)),
            "satellite_2_name":      str(row.get("object_name_2", n2)),
            "miss_distance_km":      _safe_float(row.get("distance_km")),
            "relative_velocity_kms": _safe_float(row.get("relative_velocity_kms")),
            "risk_level":            str(row.get("risk_level", "LOW")),
            "collision_probability": _safe_float(row.get("collision_probability")),
            "predicted_time":        str(row.get("detection_timestamp", "")),
            "collision_type":        str(row.get("collision_type", "")),
            "is_active":             True,
            "norad_id_1":            n1,
            "norad_id_2":            n2,
            "distance_km":           _safe_float(row.get("distance_km")),
            "type_1":                str(row.get("type_1", "UNKNOWN")),
            "type_2":                str(row.get("type_2", "UNKNOWN")),
            "altitude_km_1":         _safe_float(row.get("altitude_km_1")),
            "altitude_km_2":         _safe_float(row.get("altitude_km_2")),
            "detection_timestamp":   str(row.get("detection_timestamp", "")),
        })

    return jsonify({
        "count":          len(records),
        "total_count":    total,
        "page":           page,
        "per_page":       per_pg,
        "total_pages":    math.ceil(total / per_pg) if per_pg else 1,
        "collisions":     records,
        "simulated_time": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/collisions/globe", methods=["GET"])
def collisions_globe():
    df = _load_collision_data()
    if df.empty:
        return jsonify({"count": 0, "collisions": []})

    risk_level = request.args.get("risk_level", "").upper()
    if risk_level and risk_level != "ALL" and "risk_level" in df.columns:
        df = df[df["risk_level"] == risk_level]

    if "distance_km" in df.columns:
        df = df.sort_values("distance_km", ascending=True)

    df = df.head(min(500, int(request.args.get("limit", 500))))

    records = []
    for _, row in df.iterrows():
        records.append({
            "norad_id_1":            str(row.get("norad_id_1", "")),
            "norad_id_2":            str(row.get("norad_id_2", "")),
            "type_1":                str(row.get("type_1", "UNKNOWN")),
            "type_2":                str(row.get("type_2", "UNKNOWN")),
            "distance_km":           _safe_float(row.get("distance_km")),
            "risk_level":            str(row.get("risk_level", "LOW")),
            "collision_probability": _safe_float(row.get("collision_probability")),
            "collision_type":        str(row.get("collision_type", "")),
            "relative_velocity_kms": _safe_float(row.get("relative_velocity_kms")),
            "detection_timestamp":   str(row.get("detection_timestamp", "")),
            "altitude_km_1":         _safe_float(row.get("altitude_km_1")),
            "altitude_km_2":         _safe_float(row.get("altitude_km_2")),
            # ECI position coords — frontend uses these to place markers on the globe
            "approach_position_x":   _safe_float(row.get("pos_x_1")),
            "approach_position_y":   _safe_float(row.get("pos_y_1")),
            "approach_position_z":   _safe_float(row.get("pos_z_1")),
        })

    return jsonify({"count": len(records), "collisions": records})


@app.route("/api/collisions/high-risk", methods=["GET"])
def high_risk_collisions():
    df = _load_collision_data()
    if df.empty:
        return jsonify({"count": 0, "high_risk_collisions": []})

    if "risk_level" in df.columns:
        df = df[df["risk_level"].isin(["CRITICAL", "HIGH", "MEDIUM"])]
    if "distance_km" in df.columns:
        df = df.sort_values("distance_km", ascending=True).head(50)

    records = []
    for _, row in df.iterrows():
        n1 = str(row.get("norad_id_1", ""))
        n2 = str(row.get("norad_id_2", ""))
        records.append({
            "satellite_1_id":        n1,
            "satellite_2_id":        n2,
            "satellite_1_name":      str(row.get("object_name_1", n1)),
            "satellite_2_name":      str(row.get("object_name_2", n2)),
            "norad_id_1":            n1,
            "norad_id_2":            n2,
            "type_1":                str(row.get("type_1", "UNKNOWN")),
            "type_2":                str(row.get("type_2", "UNKNOWN")),
            "distance_km":           _safe_float(row.get("distance_km")),
            "miss_distance_km":      _safe_float(row.get("distance_km")),
            "risk_level":            str(row.get("risk_level", "LOW")),
            "collision_type":        str(row.get("collision_type", "")),
            "collision_probability": _safe_float(row.get("collision_probability")),
        })

    return jsonify({"count": len(records), "high_risk_collisions": records})


@app.route("/api/collisions/frequency", methods=["GET"])
def collisions_frequency():
    df = _load_collision_data()
    if df.empty or "norad_id_1" not in df.columns:
        return jsonify({"count": 0, "pairs": []})

    limit = min(100, int(request.args.get("limit", 20)))
    grp = (
        df.groupby(["norad_id_1", "norad_id_2"])
        .agg(
            collision_count=("distance_km", "count"),
            min_distance_km=("distance_km", "min"),
            avg_distance_km=("distance_km", "mean"),
            max_distance_km=("distance_km", "max"),
        )
        .reset_index()
        .sort_values(["collision_count", "min_distance_km"], ascending=[False, True])
        .head(limit)
    )

    pairs = []
    for _, row in grp.iterrows():
        n1 = str(row["norad_id_1"])
        n2 = str(row["norad_id_2"])
        pairs.append({
            "satellite_1_id":  n1,
            "satellite_2_id":  n2,
            "approach_events": int(row["collision_count"]),
            "collision_count": int(row["collision_count"]),
            "min_distance_km": _safe_float(row["min_distance_km"]),
            "avg_distance_km": _safe_float(row["avg_distance_km"]),
            "max_distance_km": _safe_float(row["max_distance_km"]),
        })

    return jsonify({"count": len(pairs), "pairs": pairs})


@app.route("/api/debris", methods=["GET"])
def get_debris():
    catalog = _load_debris_catalog()
    if catalog.empty:
        return jsonify({"count": 0, "debris": []})

    limit   = min(5000, int(request.args.get("limit", 1000)))
    catalog = catalog.head(limit)

    records = []
    for _, row in catalog.iterrows():
        records.append({
            "norad_id":    str(row.get("NORAD_CAT_ID", row.get("norad_id", ""))),
            "name":        str(row.get("OBJECT_NAME",  row.get("name",  ""))),
            "country":     str(row.get("COUNTRY",      "")),
            "launch":      str(row.get("LAUNCH",       "")),
            "period":      _safe_float(row.get("PERIOD",      row.get("period"))),
            "inclination": _safe_float(row.get("INCLINATION", row.get("inclination"))),
            "apogee":      _safe_float(row.get("APOGEE",      row.get("apogee"))),
            "perigee":     _safe_float(row.get("PERIGEE",     row.get("perigee"))),
            "rcs_size":    str(row.get("RCS_SIZE",    "")),
            "object_type": str(row.get("OBJECT_TYPE", row.get("object_type", ""))),
        })

    return jsonify({"count": len(records), "debris": records})


@app.route("/api/debris/<norad_id>", methods=["GET"])
def get_debris_by_id(norad_id: str):
    catalog = _load_debris_catalog()
    if catalog.empty:
        return jsonify({"error": "Catalog not available — run the pipeline first."}), 404

    id_col = "NORAD_CAT_ID" if "NORAD_CAT_ID" in catalog.columns else "norad_id"
    row = catalog[catalog[id_col].astype(str) == norad_id]
    if row.empty:
        return jsonify({"error": f"NORAD ID {norad_id} not found"}), 404

    r = row.iloc[0].to_dict()
    return jsonify({k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in r.items()})


# ─── ML + Pipeline endpoints ─────────────────────────────────────────────────

@app.route("/api/ml/metrics", methods=["GET"])
def ml_metrics():
    """Return ML model benchmark metrics from the TSA pipeline."""
    state = _load_dashboard_state()
    ml = state.get("ml_metrics", {})
    if ml:
        return jsonify({"status": "ok", "source": "tsa_pipeline", "metrics": ml})

    forecast = _load_forecast_results()
    if forecast:
        return jsonify({"status": "ok", "source": "forecast_results", "metrics": forecast})

    return jsonify({"status": "no_data",
                    "message": "Run scripts/models/run_pipeline.py first.",
                    "metrics": {}})


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    """Return the last pipeline run metadata."""
    state = _load_dashboard_state()
    if not state:
        return jsonify({"status": "no_runs",
                        "message": "Run scripts/models/run_pipeline.py first.",
                        "runs": []})

    meta = state.get("pipeline_meta", {})
    return jsonify({
        "status":     "ok",
        "last_run":   meta.get("generated_at", "unknown"),
        "total_runs": 1,
        "runs":       [meta],
    })


@app.route("/api/simulation/time", methods=["GET"])
def simulation_time():
    return jsonify({
        "current_simulated_time": datetime.utcnow().isoformat() + "Z",
        "elapsed_simulated_days": 0,
        "simulation_mode":        "real-time",
    })


# ─── TSA-specific endpoints ───────────────────────────────────────────────────

@app.route("/api/tsa/state", methods=["GET"])
def tsa_state():
    """Return the full consolidated TSA pipeline state."""
    state = _load_dashboard_state()
    if state:
        return jsonify(state)
    return _no_data_yet()


@app.route("/api/tsa/forecast", methods=["GET"])
def tsa_forecast():
    """Return per-model forecast results and benchmark table."""
    data = _load_forecast_results()
    if data:
        return jsonify(data)
    return _no_data_yet()


@app.route("/api/tsa/conjunctions", methods=["GET"])
def tsa_conjunctions():
    """Return the conjunction/close-approach report with optional risk_level filter."""
    data = _load_conjunction_report()
    if not data:
        return _no_data_yet()

    risk_filter = request.args.get("risk_level", "").upper()
    if risk_filter and risk_filter != "ALL":
        data = dict(data)
        data["alerts"] = [a for a in data.get("alerts", [])
                          if a.get("risk_level") == risk_filter]
        data["n_alerts"] = len(data["alerts"])

    return jsonify(data)


# ─── Trajectory endpoints ─────────────────────────────────────────────────────

_TS_DF_CACHE: "pd.DataFrame | None" = None

def _load_ts_df() -> "pd.DataFrame":
    """Load (and cache) the ts_df.parquet trajectory file."""
    global _TS_DF_CACHE
    if _TS_DF_CACHE is not None:
        return _TS_DF_CACHE
    if not HAS_PANDAS or not TS_DF_PARQUET.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(str(TS_DF_PARQUET))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _TS_DF_CACHE = df
        logger.info(f"ts_df.parquet loaded: {len(df):,} rows, {df['norad_id'].nunique()} objects")
        return df
    except Exception as e:
        logger.warning(f"Could not load ts_df.parquet: {e}")
        return pd.DataFrame()


def _eci_to_geodetic(x, y, z):
    """ECI (km) → geodetic lat(deg), lon(deg), alt(km)."""
    import math as _m
    r   = _m.sqrt(x*x + y*y + z*z)
    lat = _m.degrees(_m.atan2(z, _m.sqrt(x*x + y*y)))
    lon = _m.degrees(_m.atan2(y, x))
    alt = r - 6371.0
    return round(lat, 5), round(lon, 5), round(alt, 3)


def _altitude_regime(alt_km: float) -> str:
    if alt_km < 2000:    return "LEO"
    elif alt_km < 8000:  return "MEO"
    elif alt_km < 36500: return "HEO"
    else:                return "GEO"


@app.route("/api/debris/trajectory", methods=["GET"])
def debris_trajectory():
    """
    Return the 72-hour ECI trajectory for one debris object.

    Query params:
      norad_id   (required) — NORAD catalogue ID
      downsample (optional, default=1) — keep every Nth point (1=all, 6=every 6th)
    """
    norad_id = request.args.get("norad_id", "")
    if not norad_id:
        return jsonify({"error": "norad_id query param required"}), 400

    downsample = max(1, int(request.args.get("downsample", 1)))

    df = _load_ts_df()
    if df.empty:
        return _no_data_yet("Run the pipeline first (step 1 produces ts_df.parquet).")

    sat = df[df["norad_id"].astype(str) == str(norad_id)]
    if sat.empty:
        return jsonify({"error": f"NORAD ID {norad_id} not found"}), 404

    sat = sat.sort_values("timestamp").iloc[::downsample]

    points = []
    for _, row in sat.iterrows():
        lat, lon, _ = _eci_to_geodetic(row["x"], row["y"], row["z"])
        points.append({
            "timestamp": row["timestamp"].isoformat(),
            "x":          round(float(row["x"]), 3),
            "y":          round(float(row["y"]), 3),
            "z":          round(float(row["z"]), 3),
            "vx":         round(float(row["vx"]), 5),
            "vy":         round(float(row["vy"]), 5),
            "vz":         round(float(row["vz"]), 5),
            "altitude_km": round(float(row["altitude_km"]), 3),
            "speed_kms":  round(float(row["speed_kms"]), 5),
            "lat":        lat,
            "lon":        lon,
        })

    avg_alt = float(sat["altitude_km"].mean())
    return jsonify({
        "norad_id":    str(norad_id),
        "n_points":    len(points),
        "regime":      _altitude_regime(avg_alt),
        "mean_alt_km": round(avg_alt, 2),
        "trajectory":  points,
    })


@app.route("/api/debris/trajectories/all", methods=["GET"])
def all_trajectories():
    """
    Return lightweight trajectory summaries (ground-track points) for all objects.
    Designed for the dashboard globe path overlay.

    Query params:
      downsample (default=12) — keep every Nth point (~72 pts per object at 12)
      regime     (optional)   — filter by LEO / MEO / HEO / GEO
      limit      (default=200) — max number of objects
    """
    downsample = max(1, int(request.args.get("downsample", 12)))
    regime_filter = request.args.get("regime", "").upper()
    limit = min(200, max(1, int(request.args.get("limit", 200))))

    df = _load_ts_df()
    if df.empty:
        return _no_data_yet("Run the pipeline first.")

    # Compute mean altitude per object once
    mean_alts = df.groupby("norad_id")["altitude_km"].mean()
    all_ids   = mean_alts.index.tolist()

    objects = []
    for nid in all_ids[:limit]:
        avg_alt = float(mean_alts[nid])
        reg     = _altitude_regime(avg_alt)
        if regime_filter and reg != regime_filter:
            continue

        sub = df[df["norad_id"] == nid].sort_values("timestamp").iloc[::downsample]
        lats, lons = [], []
        for _, row in sub.iterrows():
            la, lo, _ = _eci_to_geodetic(row["x"], row["y"], row["z"])
            lats.append(la)
            lons.append(lo)

        objects.append({
            "norad_id":    int(nid),
            "regime":      reg,
            "mean_alt_km": round(avg_alt, 2),
            "lats":        lats,
            "lons":        lons,
            "n_points":    len(lats),
        })

    return jsonify({
        "count":   len(objects),
        "objects": objects,
    })


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_API_PORT", 5050))
    logger.info(f"Space Debris Dashboard API  →  http://0.0.0.0:{port}")
    logger.info(f"  Output dir : {OUTPUT_DIR}")
    logger.info(f"  Pipeline   : python scripts/models/run_pipeline.py")
    app.run(host="0.0.0.0", port=port, debug=False)
