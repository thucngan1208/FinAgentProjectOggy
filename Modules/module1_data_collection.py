"""
FinAgent - Module 1: Data Collection
=====================================
Automatically collects financial data from multiple sources:
  - Stock prices (via yfinance)
  - Financial statements (via yfinance)
  - News & sentiment (via NewsAPI)
  - Macro indicators: exchange rates & commodity prices (via Alpha Vantage)

Setup:
    pip install yfinance requests python-dotenv schedule pandas

Environment variables (.env):
    NEWS_API_KEY=your_newsapi_key
    ALPHA_VANTAGE_KEY=your_alpha_vantage_key
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import requests
import pandas as pd
from dotenv import load_dotenv

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

NEWS_API_KEY       = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY", "")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["AAPL", "MSFT", "GOOGL", "NVDA"]   # stocks to track
PERIOD = "1y"                                  # lookback for price history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helper: retry wrapper ─────────────────────────────────────────────────────

def retry(func, retries: int = 3, delay: float = 2.0):
    """Call func up to `retries` times with exponential back-off."""
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            log.warning(f"Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(delay * attempt)
    log.error(f"All {retries} attempts failed for {func.__name__}")
    return None


# ── Source 1: Stock Prices (yfinance) ────────────────────────────────────────

def collect_stock_prices(tickers: list[str] = ASSETS, period: str = PERIOD) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV price history for each ticker via yfinance.

    Returns:
        dict mapping ticker → DataFrame with columns [Open, High, Low, Close, Volume, Adj Close]
    """
    results = {}
    for ticker in tickers:
        log.info(f"[Stock Prices] Fetching {ticker} ...")
        def fetch():
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, auto_adjust=False)
            if hist.empty:
                raise ValueError(f"Empty response for {ticker}")
            return hist

        df = retry(fetch)
        if df is not None:
            path = RAW_DIR / f"prices_{ticker}.csv"
            df.to_csv(path)
            log.info(f"  ✓ Saved {len(df)} rows → {path}")
            results[ticker] = df
        else:
            log.error(f"  ✗ Could not fetch prices for {ticker}")

    return results


# ── Source 2: Financial Statements (yfinance) ────────────────────────────────

def collect_financial_statements(tickers: list[str] = ASSETS) -> dict[str, dict]:
    """
    Fetch quarterly income statement, balance sheet, and cash flow
    for each ticker via yfinance.

    Returns:
        dict mapping ticker → {"income": df, "balance": df, "cashflow": df}
    """
    results = {}
    for ticker in tickers:
        log.info(f"[Financials] Fetching statements for {ticker} ...")
        def fetch():
            tk = yf.Ticker(ticker)
            return {
                "income":    tk.quarterly_financials.T,   # transpose → dates as rows
                "balance":   tk.quarterly_balance_sheet.T,
                "cashflow":  tk.quarterly_cashflow.T,
            }

        statements = retry(fetch)
        if statements:
            for stmt_name, df in statements.items():
                path = RAW_DIR / f"stmt_{ticker}_{stmt_name}.csv"
                df.to_csv(path)
                log.info(f"  ✓ Saved {stmt_name} ({len(df)} periods) → {path}")
            results[ticker] = statements
        else:
            log.error(f"  ✗ Could not fetch statements for {ticker}")

    return results


# ── Source 3: News & Sentiment (NewsAPI) ─────────────────────────────────────

def collect_news(query: str = "stock market financial", page_size: int = 50) -> pd.DataFrame:
    """
    Fetch recent financial news headlines from NewsAPI.

    Requires:
        NEWS_API_KEY env var  (free tier: https://newsapi.org)

    Returns:
        DataFrame with columns [publishedAt, source, title, description, url]
    """
    if not NEWS_API_KEY:
        log.warning("[News] NEWS_API_KEY not set — skipping news collection.")
        return pd.DataFrame()

    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "from":     (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "apiKey":   NEWS_API_KEY,
    }

    log.info(f"[News] Fetching headlines for query='{query}' ...")

    def fetch():
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise ValueError(f"API error: {data.get('message')}")
        return data["articles"]

    articles = retry(fetch)
    if not articles:
        return pd.DataFrame()

    rows = [
        {
            "publishedAt": a["publishedAt"],
            "source":      a["source"]["name"],
            "title":       a["title"],
            "description": a.get("description", ""),
            "url":         a["url"],
        }
        for a in articles
    ]
    df = pd.DataFrame(rows)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])

    path = RAW_DIR / "news_headlines.csv"
    df.to_csv(path, index=False)
    log.info(f"  ✓ Saved {len(df)} articles → {path}")
    return df

# ── Source 4a: Exchange Rates (Alpha Vantage) ────────────────────────────────

def collect_exchange_rates(pairs: list[tuple] = [("USD", "EUR"), ("USD", "JPY"), ("USD", "VND")]) -> dict[str, pd.DataFrame]:
    """
    Download daily FX rates from Alpha Vantage.

    Requires:
        ALPHA_VANTAGE_KEY env var  (free tier: https://www.alphavantage.co)

    Returns:
        dict mapping "FROM/TO" → DataFrame with OHLC columns
    """
    if not ALPHA_VANTAGE_KEY:
        log.warning("[FX] ALPHA_VANTAGE_KEY not set — skipping FX rates.")
        return {}

    base_url = "https://www.alphavantage.co/query"
    results = {}

    for from_sym, to_sym in pairs:
        label = f"{from_sym}/{to_sym}"
        log.info(f"[FX] Fetching {label} ...")

        def fetch(f=from_sym, t=to_sym):
            params = {
                "function":     "FX_DAILY",
                "from_symbol":  f,
                "to_symbol":    t,
                "outputsize":   "compact",   # last 100 data points
                "apikey":       ALPHA_VANTAGE_KEY,
            }
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            ts = data.get("Time Series FX (Daily)", {})
            if not ts:
                raise ValueError(f"No data returned: {data.get('Information', data)}")
            df = pd.DataFrame(ts).T.rename(columns={
                "1. open": "Open", "2. high": "High",
                "3. low":  "Low",  "4. close": "Close",
            }).astype(float)
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            return df

        df = retry(fetch)
        if df is not None:
            path = RAW_DIR / f"fx_{from_sym}_{to_sym}.csv"
            df.to_csv(path)
            log.info(f"  ✓ Saved {len(df)} rows → {path}")
            results[label] = df
            time.sleep(12)   # Alpha Vantage free tier: 5 calls/min
        else:
            log.error(f"  ✗ Could not fetch {label}")

    return results


# ── Source 4b: Commodity Prices (yfinance) ───────────────────────────────────

def collect_commodity_prices(period: str = PERIOD) -> dict[str, pd.DataFrame]:
    """
    Download gold (GC=F) and oil (CL=F) futures prices via yfinance.

    Returns:
        dict mapping commodity name → DataFrame
    """
    commodities = {"Gold": "GC=F", "Oil (WTI)": "CL=F"}
    results = {}

    for name, symbol in commodities.items():
        log.info(f"[Commodities] Fetching {name} ({symbol}) ...")

        def fetch(sym=symbol):
            df = yf.download(sym, period=period, progress=False, auto_adjust=False)
            if df.empty:
                raise ValueError("Empty response")
            return df

        df = retry(fetch)
        if df is not None:
            path = RAW_DIR / f"commodity_{name.replace(' ', '_')}.csv"
            df.to_csv(path)
            log.info(f"  ✓ Saved {len(df)} rows → {path}")
            results[name] = df
        else:
            log.error(f"  ✗ Could not fetch {name}")

    return results


# ── Manifest: log what was collected ─────────────────────────────────────────

def save_manifest(summary: dict) -> None:
    """Write a JSON manifest of collected files with timestamps."""
    manifest = {
        "collected_at": datetime.now().isoformat(),
        "sources": summary,
    }
    path = RAW_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    log.info(f"Manifest saved → {path}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_collection() -> dict:
    """Run all collection modules and return a summary dict."""
    log.info("=" * 60)
    log.info("FinAgent Data Collection — started")
    log.info("=" * 60)

    summary = {}

    prices     = collect_stock_prices()
    summary["stock_prices"] = list(prices.keys())

    statements = collect_financial_statements()
    summary["financial_statements"] = list(statements.keys())

    news       = collect_news()
    summary["news_articles"] = len(news) if not news.empty else 0

    fx         = collect_exchange_rates()
    summary["exchange_rates"] = list(fx.keys())

    commodities = collect_commodity_prices()
    summary["commodities"] = list(commodities.keys())

    save_manifest(summary)

    log.info("=" * 60)
    log.info(f"Collection complete. Summary: {summary}")
    log.info("=" * 60)

    return {
        "prices":      prices,
        "statements":  statements,
        "news":        news,
        "fx":          fx,
        "commodities": commodities,
    }


if __name__ == "__main__":
    run_collection()
