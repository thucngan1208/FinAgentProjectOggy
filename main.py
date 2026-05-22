"""
FinAgent - Main Pipeline Runner
================================
Runs all four modules in sequence:
  1. Data Collection
  2. Data Cleaning & Processing
  3. Visualization
  4. AI Analysis (Google Gemini)

Usage:
    python main.py                  # full pipeline, default tickers
    python main.py --tickers AAPL MSFT NVDA
    python main.py --skip-collection  # if raw data already exists
"""

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("FinAgent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="FinAgent — AI Financial Data Pipeline")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "NVDA"],
                   help="Stock tickers to analyse")
    p.add_argument("--skip-collection",  action="store_true", help="Skip data collection (use existing raw data)")
    p.add_argument("--skip-cleaning",    action="store_true", help="Skip cleaning (use existing clean data)")
    p.add_argument("--skip-viz",         action="store_true", help="Skip visualization")
    p.add_argument("--skip-ai",          action="store_true", help="Skip AI analysis")
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"FinAgent starting — tickers: {args.tickers}")

    # ── Module 1: Data Collection ─────────────────────────────────────────────
    if not args.skip_collection:
        log.info("\n" + "="*60)
        log.info("STEP 1 — Data Collection")
        log.info("="*60)
        from Modules.module1_data_collection import run_collection, ASSETS
        ASSETS.clear()
        ASSETS.extend(args.tickers)
        run_collection()
    else:
        log.info("Skipping data collection.")

    # ── Module 2: Data Cleaning ───────────────────────────────────────────────
    if not args.skip_cleaning:
        log.info("\n" + "="*60)
        log.info("STEP 2 — Data Cleaning & Feature Engineering")
        log.info("="*60)
        from Modules.module2_data_cleaning import run_cleaning
        run_cleaning(tickers=args.tickers)
    else:
        log.info("Skipping data cleaning.")

    # ── Module 3: Visualization ───────────────────────────────────────────────
    if not args.skip_viz:
        log.info("\n" + "="*60)
        log.info("STEP 3 — Visualization")
        log.info("="*60)
        from Modules.module3_visualization import run_visualization
        chart_registry = run_visualization(tickers=args.tickers)
        log.info(f"Generated {len(chart_registry)} charts.")
    else:
        log.info("Skipping visualization.")

    # ── Module 4: AI Analysis ─────────────────────────────────────────────────
    if not args.skip_ai:
        log.info("\n" + "="*60)
        log.info("STEP 4 — AI Analysis (Google Gemini)")
        log.info("="*60)
        from Modules.module4_ai_analysis import run_ai_analysis
        report_path = run_ai_analysis(tickers=args.tickers)
        abs_path = Path(report_path).resolve()
        log.info(f"\n✅ Pipeline complete!")
        log.info(f"   HTML Report: {abs_path}")
        log.info("   Open in your browser to view all charts + AI commentary.")
    else:
        log.info("Skipping AI analysis.")

    log.info("\nFinAgent pipeline finished successfully.")


if __name__ == "__main__":
    main()
