"""
06_continuous_detection.py
============================
Continuous collision detection — feeds every historical TLE year file
(2004 → 2025) through the data pipeline + conjunction assessment + dashboard
update in sequence, building up a cumulative collision_history.parquet.

How it works
------------
Each "cycle" processes ONE TLE year file:
  1. Run step 1  → parse & SGP4-densify that year's TLEs → ts_df.parquet
  4. Run step 4  → conjunction screening           → collision_alerts_<ts>/
                                                     conjunction_report.json
                                                     collision_history.parquet  (APPEND)
  5. Run step 5  → rebuild dashboard_state.json    → dashboard refreshes live

After all year files are processed the loop can either:
  - stop (default / --once)
  - repeat indefinitely with a configurable sleep (--loop)

Usage examples
--------------
  # Process all years once, then stop:
  python scripts/models/06_continuous_detection.py

  # Process only 2022-2025 once:
  python scripts/models/06_continuous_detection.py --years 2022 2023 2024 2025

  # Process all years then repeat every 10 minutes forever:
  python scripts/models/06_continuous_detection.py --loop --interval 600

  # Dry-run (no actual computation):
  python scripts/models/06_continuous_detection.py --dry-run

  # Wider screening radius (default 50 km):
  python scripts/models/06_continuous_detection.py --screen-km 100
"""

import argparse, importlib.util, logging, time, sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("continuous")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT        = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "models"
RAW_DIR     = ROOT / "data" / "raw"
OUTPUT_DIR  = ROOT / "Output"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Ordered list of all available TLE year files (filename → display label)
# Multi-part 2004 files are merged as a single logical year
ALL_TLE_FILES = [
    "tle2004_1of8.txt",
    "tle2004_2of8.txt",
    "tle2004_3of8.txt",
    "tle2004_4of8.txt",
    "tle2004_5of8.txt",
    "tle2005.txt",
    "tle2006.txt",
    "tle2007.txt",
    "tle2008.txt",
    "tle2009.txt",
    "tle2010.txt",
    "tle2011.txt",
    "tle2012_.txt",
    "tle2013.txt",
    "tle2014_.txt",
    "tle2015_.txt",
    "tle2016.txt",
    "tle2017.txt",
    "tle2018.txt",
    "tle2019.txt",
    "tle2020.txt",
    "tle2021.txt",
    "tle2022.txt",
    "tle2023_.txt",
    "tle2024.txt",
    "tle2025_.txt",
]


def _import_step(filename: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _banner(text: str):
    log.info("━" * 64)
    log.info(f"  {text}")
    log.info("━" * 64)


def _process_one_file(
    tle_file: str,
    file_idx: int,
    total_files: int,
    out_dir: Path,
    args,
    dry_run: bool,
):
    """Run steps 1 → 4 → 5 for a single TLE file."""
    tle_path = RAW_DIR / tle_file
    if not tle_path.exists():
        log.warning(f"  ⚠ TLE file not found, skipping: {tle_path}")
        return False

    _banner(f"[{file_idx}/{total_files}] Processing: {tle_file}")
    cycle_start = time.time()

    # ── Step 1: Data pipeline ─────────────────────────────────────────────
    log.info(f"  ▶ Step 1/3 — SGP4 densification ({tle_file})")
    if not dry_run:
        dp = _import_step("01_data_pipeline.py")
        t0 = time.time()
        dp.run(
            tle_file=tle_file,
            max_records=args.max_records,
            top_n=args.top_n,
            resample_min=args.resample_min,
            propagate_hrs=args.propagate_hrs,
            min_span_hrs=args.min_span_hrs,
            output_dir=out_dir,
        )
        log.info(f"     ✅ Data pipeline done in {time.time()-t0:.1f}s")
    else:
        log.info("     [DRY RUN] skipped")

    # ── Step 4: Conjunction assessment (append to history) ────────────────
    log.info(f"  ▶ Step 2/3 — Conjunction screening ({tle_file})")
    if not dry_run:
        ca = _import_step("04_conjunction_assessment.py")
        t0 = time.time()
        alerts_df = ca.run(
            output_dir=out_dir,
            screen_km=args.screen_km,
            screen_hours=args.screen_hours,
            tle_file=tle_file,
            append_history=True,          # ← always accumulate
        )
        n_alerts = len(alerts_df) if alerts_df is not None else 0
        log.info(f"     ✅ Conjunction done in {time.time()-t0:.1f}s  "
                 f"| {n_alerts} new alerts")
    else:
        log.info("     [DRY RUN] skipped")

    # ── Step 5: Dashboard update ──────────────────────────────────────────
    log.info(f"  ▶ Step 3/3 — Dashboard update")
    if not dry_run:
        du = _import_step("05_update_dashboard.py")
        t0 = time.time()
        du.run(output_dir=out_dir)
        log.info(f"     ✅ Dashboard updated in {time.time()-t0:.1f}s")
    else:
        log.info("     [DRY RUN] skipped")

    elapsed = time.time() - cycle_start
    log.info(f"  ✅ {tle_file} complete in {elapsed:.1f}s")

    # Print running history size
    history_path = out_dir / "collision_history.parquet"
    if history_path.exists() and not dry_run:
        import pandas as pd
        hist = pd.read_parquet(history_path)
        log.info(f"  📊 Cumulative history: {len(hist):,} total collision events")

    return True


def _year_from_filename(fname: str) -> int:
    """Extract the 4-digit year from a TLE filename."""
    for part in fname.split(".")[0].split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Space Debris TSA — Continuous Collision Detection")
    parser.add_argument(
        "--years", nargs="+", type=int, default=None, metavar="YYYY",
        help="Process only these years (e.g. --years 2020 2021 2022). "
             "Default: all available years.")
    parser.add_argument(
        "--loop", action="store_true",
        help="After finishing all files, loop back and repeat indefinitely.")
    parser.add_argument(
        "--interval", type=float, default=60.0, metavar="SEC",
        help="Seconds to wait between full cycles when --loop is set (default: 60).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing.")
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Output directory (default: Output/)")
    parser.add_argument(
        "--max-records", type=int, default=150_000)
    parser.add_argument(
        "--top-n",       type=int, default=200,
        help="Objects to keep per TLE file (default: 200)")
    parser.add_argument(
        "--resample-min", type=int, default=5,
        help="SGP4 propagation step in minutes (default: 5)")
    parser.add_argument(
        "--propagate-hrs", type=float, default=72.0,
        help="Hours to propagate each TLE (default: 72)")
    parser.add_argument(
        "--min-span-hrs", type=float, default=12.0,
        help="Min hours span to keep an object (default: 12)")
    parser.add_argument(
        "--screen-km",    type=float, default=50.0,
        help="Conjunction screening distance in km (default: 50)")
    parser.add_argument(
        "--screen-hours", type=float, default=24.0,
        help="Screening time window in hours (default: 24)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter TLE files by requested years
    if args.years:
        year_set = set(args.years)
        tle_files = [f for f in ALL_TLE_FILES
                     if _year_from_filename(f) in year_set]
        if not tle_files:
            log.error(f"No TLE files found for years: {args.years}")
            sys.exit(1)
    else:
        tle_files = [f for f in ALL_TLE_FILES if (RAW_DIR / f).exists()]

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  CONTINUOUS COLLISION DETECTION — STARTING              ║")
    log.info(f"║  TLE files : {len(tle_files):<44}║")
    log.info(f"║  Screen km : {args.screen_km:<44}║")
    log.info(f"║  Loop mode : {'YES (interval=' + str(args.interval) + 's)' if args.loop else 'NO (single pass)':<44}║")
    log.info(f"║  Output    : {str(out_dir):<44}║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    cycle_num = 0
    total_start = time.time()

    while True:
        cycle_num += 1
        cycle_label = f"Cycle {cycle_num}" if args.loop else "Single pass"
        log.info(f"\n{'═'*64}")
        log.info(f"  🔄  {cycle_label} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        log.info(f"{'═'*64}\n")

        for idx, tle_file in enumerate(tle_files, start=1):
            _process_one_file(
                tle_file=tle_file,
                file_idx=idx,
                total_files=len(tle_files),
                out_dir=out_dir,
                args=args,
                dry_run=args.dry_run,
            )

        total_elapsed = time.time() - total_start
        log.info(f"\n{'═'*64}")
        log.info(f"  ✅  {cycle_label} complete — total elapsed {total_elapsed:.1f}s")

        # Print final history summary
        history_path = out_dir / "collision_history.parquet"
        if history_path.exists() and not args.dry_run:
            import pandas as pd
            hist = pd.read_parquet(history_path)
            risk_counts = hist["risk_level"].value_counts().to_dict() if "risk_level" in hist.columns else {}
            log.info(f"  📊 Final history: {len(hist):,} events | "
                     f"CRITICAL={risk_counts.get('CRITICAL', 0)}  "
                     f"HIGH={risk_counts.get('HIGH', 0)}  "
                     f"MEDIUM={risk_counts.get('MEDIUM', 0)}  "
                     f"LOW={risk_counts.get('LOW', 0)}")
        log.info(f"{'═'*64}\n")

        if not args.loop:
            break

        log.info(f"  ⏳  Next cycle in {args.interval:.0f}s "
                 f"— press Ctrl-C to stop")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("  ⛔  Interrupted by user. Exiting.")
            break

    log.info("Continuous detection finished.")


if __name__ == "__main__":
    main()
