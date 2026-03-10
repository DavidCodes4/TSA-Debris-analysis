"""
run_pipeline.py
================
End-to-end orchestrator — runs all 5 pipeline steps in sequence.

Usage:
  python scripts/models/run_pipeline.py                # all steps, tle2024.txt
  python scripts/models/run_pipeline.py --steps 1 2    # only step 1 & 2
  python scripts/models/run_pipeline.py --steps 4 5    # skip training
  python scripts/models/run_pipeline.py --tle-all      # run steps 4+5 on every year
  python scripts/models/run_pipeline.py --continuous   # continuous loop over all years
  python scripts/models/run_pipeline.py --help

Steps
-----
  1  Data pipeline      (TLE → SGP4 densification → ts_df.parquet)
  2  Model training     (VAR + LSTM + PatchTST → saved .keras/.pkl)
  3  Evaluation         (benchmark all models → forecast_results.json)
  4  Conjunction        (multi-object risk screening → collision_alerts_*.csv)
  5  Dashboard update   (consolidate → dashboard_state.json + catalog.csv)
"""

import argparse, logging, sys, time, importlib.util
from pathlib import Path

log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")

# ── Path setup — works from any working directory ─────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "models"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import_step(filename: str):
    """Import a pipeline step module by filename (e.g. '01_data_pipeline.py')."""
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(n: int, label: str, fn, dry_run: bool = False, **kwargs):
    log.info("━" * 60)
    log.info(f"  STEP {n}/5 — {label}")
    log.info("━" * 60)
    if dry_run:
        log.info(f"  [DRY RUN] skipping execution")
        return None
    t0 = time.time()
    result = fn(**kwargs)
    elapsed = time.time() - t0
    log.info(f"  ✅ Step {n} complete in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Space Debris TSA — Full Pipeline Orchestrator")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing")
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        metavar="N",
        help="Pipeline steps to run (default: 1 2 3 4 5)")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "Output"),
        help="Output directory (default: Output/)")
    parser.add_argument(
        "--tle-file",    default="tle2024.txt",
        help="TLE filename inside data/raw/, or 'ALL' to use every year file")
    parser.add_argument(
        "--tle-all-data", action="store_true",
        help="Shorthand for --tle-file ALL — load all years for step 1")
    parser.add_argument(
        "--tle-all", action="store_true",
        help="Run steps 4+5 over every available TLE year file "
             "(equivalent to running 06_continuous_detection.py --once)")
    parser.add_argument(
        "--continuous", action="store_true",
        help="Hand off to 06_continuous_detection.py — loop all years forever")
    parser.add_argument(
        "--continuous-interval", type=float, default=60.0, metavar="SEC",
        help="Seconds between loops when --continuous is set (default: 60)")
    parser.add_argument(
        "--max-records", type=int, default=150_000,
        help="Max TLE lines to read (default: 150000)")
    parser.add_argument(
        "--top-n",       type=int, default=200,
        help="Number of objects to keep (default: 200)")
    parser.add_argument(
        "--resample-min",type=int, default=5,
        help="SGP4 propagation step in minutes (default: 5)")
    parser.add_argument(
        "--propagate-hrs",type=float, default=72.0,
        help="Hours to propagate each TLE (default: 72)")
    parser.add_argument(
        "--min-span-hrs", type=float, default=12.0,
        help="Min hours span required to keep object (default: 12)")
    parser.add_argument(
        "--sample-id",   type=int, default=25544,
        help="NORAD ID to use for single-object model training (default: 25544 ISS)")
    parser.add_argument(
        "--screen-km",   type=float, default=50.0,
        help="Conjunction screening distance in km (default: 50)")
    parser.add_argument(
        "--screen-hours",type=float, default=24.0,
        help="Conjunction screening time window in hours (default: 24)")
    args = parser.parse_args()

    # ── --continuous: delegate entirely to 06_continuous_detection.py ────────
    if args.continuous:
        log.info("Handing off to continuous detection loop …")
        cd = _import_step("06_continuous_detection.py")
        # Monkey-patch sys.argv so argparse inside the module gets the right args
        import sys as _sys
        _sys.argv = [
            "06_continuous_detection.py",
            "--loop",
            "--interval", str(args.continuous_interval),
            "--output-dir", args.output_dir,
            "--screen-km",  str(args.screen_km),
            "--screen-hours", str(args.screen_hours),
            "--top-n",      str(args.top_n),
            "--max-records", str(args.max_records),
            "--resample-min", str(args.resample_min),
            "--propagate-hrs", str(args.propagate_hrs),
        ] + (["--dry-run"] if args.dry_run else [])
        cd.main()
        return

    # ── --tle-all: run steps 4+5 for every year (no retraining) ──────────────
    if args.tle_all:
        cd = _import_step("06_continuous_detection.py")
        import sys as _sys
        _sys.argv = [
            "06_continuous_detection.py",
            "--output-dir", args.output_dir,
            "--screen-km",  str(args.screen_km),
            "--screen-hours", str(args.screen_hours),
            "--top-n",      str(args.top_n),
            "--max-records", str(args.max_records),
            "--resample-min", str(args.resample_min),
            "--propagate-hrs", str(args.propagate_hrs),
        ] + (["--dry-run"] if args.dry_run else [])
        cd.main()
        return

    steps    = sorted(set(args.steps))
    out_dir  = Path(args.output_dir)
    dry_run  = args.dry_run
    out_dir.mkdir(parents=True, exist_ok=True)

    # --tle-all-data is a shorthand for --tle-file ALL
    if args.tle_all_data:
        args.tle_file = "ALL"

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   SPACE DEBRIS TSA — PIPELINE STARTING              ║")
    log.info(f"║   Steps to run: {steps}                               ")
    log.info(f"║   Output dir  : {out_dir}                             ")
    log.info("╚══════════════════════════════════════════════════════╝")

    t_total = time.time()

    # ── Step 1: Data pipeline ─────────────────────────────────────────────────
    if 1 in steps:
        dp = _import_step("01_data_pipeline.py")
        _step(1, "Data Pipeline (SGP4 densification)", dp.run,
              dry_run=dry_run,
              tle_file=args.tle_file,
              max_records=args.max_records,
              top_n=args.top_n,
              resample_min=args.resample_min,
              propagate_hrs=args.propagate_hrs,
              min_span_hrs=args.min_span_hrs,
              output_dir=out_dir)

    # ── Step 2: Model training ────────────────────────────────────────────────
    if 2 in steps:
        tm = _import_step("02_train_models.py")
        _step(2, "Model Training (VAR + LSTM + PatchTST)", tm.run,
              dry_run=dry_run,
              sample_id=args.sample_id,
              output_dir=out_dir)

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    if 3 in steps:
        ev = _import_step("03_evaluate_models.py")
        _step(3, "Evaluation & Benchmarking", ev.run,
              dry_run=dry_run,
              output_dir=out_dir)

    # ── Step 4: Conjunction assessment ───────────────────────────────────────
    if 4 in steps:
        ca = _import_step("04_conjunction_assessment.py")
        _step(4, "Conjunction Assessment", ca.run,
              dry_run=dry_run,
              output_dir=out_dir,
              screen_km=args.screen_km,
              screen_hours=args.screen_hours,
              tle_file=args.tle_file,
              append_history=True)

    # ── Step 5: Dashboard update ──────────────────────────────────────────────
    if 5 in steps:
        du = _import_step("05_update_dashboard.py")
        _step(5, "Dashboard Update", du.run,
              dry_run=dry_run,
              output_dir=out_dir)

    elapsed = time.time() - t_total
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info(f"║  ✅ PIPELINE COMPLETE in {elapsed:.1f}s               ")
    log.info(f"║  Output: {out_dir}                                    ")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("║  Next: start the Flask API:                          ║")
    log.info("║    python dashboard_api.py                           ║")
    log.info("║  Then start the React dashboard:                     ║")
    log.info("║    cd dashboard && npm run dev                       ║")
    log.info("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
