"""
FinAgent - Module 2: Data Cleaning & Processing
================================================
Cleans raw financial data collected by module1_data_collection.py.

Setup:
    pip install pandas numpy scipy
"""

import logging
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

RAW_DIR    = Path("data/raw")
CLEAN_DIR  = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

OUTLIER_ZSCORE_THRESHOLD = 3.5
IQR_MULTIPLIER           = 3.0
ROLLING_WINDOWS          = [7, 30]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/cleaning.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
cleaning_report: dict = {}


def _record(section, key, value):
    cleaning_report.setdefault(section, {})[key] = value


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_raw_prices(ticker: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"prices_{ticker}.csv"
    if not path.exists():
        log.warning(f"[Load] {path} not found — skipping.")
        return None
    # Read without parse_dates first to inspect
    df = pd.read_csv(path, index_col=0)
    log.info(f"[Load] {ticker}: {len(df)} rows, index sample: {df.index[:3].tolist()}")
    return df


# ── Step 2: Normalise ─────────────────────────────────────────────────────────

def normalise_types(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == '' or c[1] == ticker else f"{c[0]}_{c[1]}"
                      for c in df.columns]

    df.columns = df.columns.str.strip()
    log.info(f"[Normalise] {ticker}: raw columns = {list(df.columns)}")

    # Rename to standard names
    rename_map = {}
    for col in df.columns:
        cu = col.upper().replace(" ", "_")
        if "ADJ" in cu and "CLOSE" in cu:
            rename_map[col] = "Adj_Close"
        elif cu == "OPEN" or cu.endswith("_OPEN"):
            rename_map[col] = "Open"
        elif cu == "HIGH" or cu.endswith("_HIGH"):
            rename_map[col] = "High"
        elif cu == "LOW" or cu.endswith("_LOW"):
            rename_map[col] = "Low"
        elif cu == "CLOSE" or cu.endswith("_CLOSE"):
            rename_map[col] = "Close"
        elif cu == "VOLUME" or cu.endswith("_VOLUME"):
            rename_map[col] = "Volume"
    df.rename(columns=rename_map, inplace=True)

    # Keep only standard columns that exist
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Parse index as dates (handle timezone)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    # Coerce to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"[Normalise] {ticker}: final columns = {list(df.columns)}, rows = {len(df)}")
    _record(ticker, "columns", list(df.columns))
    return df


# ── Step 3: Remove duplicates ─────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    dupes = df.index.duplicated().sum()
    if dupes:
        log.warning(f"[Duplicates] {ticker}: {dupes} duplicate dates removed.")
        df = df[~df.index.duplicated(keep="last")]
    _record(ticker, "duplicates_removed", int(dupes))
    return df


# ── Step 4: Handle missing values (simple, no reindex) ───────────────────────

def handle_missing(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing_before = int(df.isna().sum().sum())

    # Forward fill then backward fill (no reindex to avoid row explosion)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    missing_after = int(df.isna().sum().sum())
    log.info(f"[Missing] {ticker}: NaN before={missing_before}, after={missing_after}, rows={len(df)}")
    _record(ticker, "missing_values", {"before": missing_before, "after": missing_after})
    return df


# ── Step 5: Outlier detection ─────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if "Close" not in df.columns:
        return df

    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    # Z-score on returns
    z = np.abs(stats.zscore(log_ret))
    z_series = pd.Series(False, index=df.index)
    z_series.loc[log_ret.index] = z > OUTLIER_ZSCORE_THRESHOLD
    df["outlier_return"] = z_series

    # IQR on price level
    Q1, Q3 = df["Close"].quantile(0.25), df["Close"].quantile(0.75)
    IQR = Q3 - Q1
    df["outlier_price"] = (
        (df["Close"] < Q1 - IQR_MULTIPLIER * IQR) |
        (df["Close"] > Q3 + IQR_MULTIPLIER * IQR)
    )

    n_ret = int(df["outlier_return"].sum())
    n_prc = int(df["outlier_price"].sum())
    log.info(f"[Outliers] {ticker}: return outliers={n_ret}, price outliers={n_prc}")
    _record(ticker, "outliers", {"return": n_ret, "price": n_prc})
    return df


# ── Step 6: Feature engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"

    df["daily_return"] = df[price_col].pct_change()
    df["log_return"]   = np.log(df[price_col] / df[price_col].shift(1))

    for w in ROLLING_WINDOWS:
        df[f"sma_{w}"]        = df[price_col].rolling(w).mean()
        df[f"ema_{w}"]        = df[price_col].ewm(span=w, adjust=False).mean()
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std() * np.sqrt(252)

    # Bollinger Bands (20-day)
    sma20 = df[price_col].rolling(20).mean()
    std20 = df[price_col].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20

    # RSI (14-day)
    delta = df[price_col].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    log.info(f"[Features] {ticker}: {len(df.columns)} total columns, {len(df)} rows")
    return df


# ── Step 7: Save ─────────────────────────────────────────────────────────────

def save_clean(df: pd.DataFrame, ticker: str) -> Path:
    path = CLEAN_DIR / f"clean_{ticker}.csv"
    df.to_csv(path)
    log.info(f"[Save] {ticker}: {len(df)} rows → {path}")
    return path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def clean_ticker(ticker: str) -> pd.DataFrame | None:
    df = load_raw_prices(ticker)
    if df is None:
        return None
    df = normalise_types(df, ticker)
    df = remove_duplicates(df, ticker)
    df = handle_missing(df, ticker)
    df = detect_outliers(df, ticker)
    df = engineer_features(df, ticker)
    save_clean(df, ticker)
    return df


def run_cleaning(tickers=None):
    log.info("=" * 60)
    log.info("FinAgent Data Cleaning — started")
    log.info("=" * 60)

    if tickers is None:
        tickers = sorted({
            p.stem.replace("prices_", "")
            for p in RAW_DIR.glob("prices_*.csv")
        })
        log.info(f"Auto-discovered tickers: {tickers}")

    results = {}
    for ticker in tickers:
        log.info(f"\n── Processing {ticker} ──")
        df = clean_ticker(ticker)
        if df is not None:
            results[ticker] = df

    report_path = CLEAN_DIR / "cleaning_report.json"
    report_path.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "tickers": cleaning_report},
                   indent=2, default=str),
        encoding="utf-8",
    )
    log.info(f"\nCleaning report → {report_path}")
    log.info("=" * 60)
    log.info(f"Cleaning complete. Processed: {list(results.keys())}")
    log.info("=" * 60)
    return results


if __name__ == "__main__":
    run_cleaning()
