# FinAgent — AI-Powered Financial Data Agent

> **Midterm Group Project** · IT Application in Banking and Finance · 2026  
> **Team:** [TEAM NAME] · **Deadline:** 27 May 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%202.0%20Flash-orange?logo=google)

FinAgent is an **end-to-end autonomous financial data pipeline** that collects market data from multiple live sources, cleans and engineers features, generates publication-quality visualisations, and produces AI-powered natural language analysis — all from a single command.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Setup and Installation](#setup-and-installation)
6. [Configuration — API Keys](#configuration--api-keys)
7. [Usage](#usage)
8. [Data Sources](#data-sources)
9. [Data Cleaning Pipeline](#data-cleaning-pipeline)
10. [Visualisations Produced](#visualisations-produced)
11. [AI Analysis Module](#ai-analysis-module)
12. [Output Files Reference](#output-files-reference)
13. [Troubleshooting](#troubleshooting)
14. [Dependencies](#dependencies)
15. [Notes and Limitations](#notes-and-limitations)

---

## Overview

FinAgent mirrors real-world **fintech data engineering workflows** by automating the full journey from raw data collection through to natural language analytical summaries. The system tracks four large-cap US technology equities — **AAPL, MSFT, GOOGL, NVDA** — over a configurable historical window, though any valid ticker supported by Yahoo Finance can be substituted.

The pipeline is structured into four independent modules, each corresponding to one stage of the data lifecycle:

| Stage | Module | Responsibility |
|---|---|---|
| 1 | `module1_data_collection.py` | Collect from 4 source types |
| 2 | `module2_data_cleaning.py` | Clean, normalise, engineer features |
| 3 | `module3_visualization.py` | Generate 4+ chart types |
| 4 | `module4_ai_analysis.py` | LLM analysis + HTML report |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FinAgent Pipeline                            │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Module 1    │    │  Module 2    │    │  Module 3    │          │
│  │   Data       │───▶│   Cleaning   │───▶│ Visualisation│          │
│  │  Collection  │    │  & Features  │    │  (4+ charts) │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                  │                   │                   │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼───────┐           │
│  │  data/raw/  │    │ data/clean/ │    │ data/charts/ │           │
│  │  (CSV, JSON)│    │ (CSV + eng.)│    │  (PNG, HTML) │           │
│  └─────────────┘    └─────────────┘    └──────┬───────┘           │
│                                               │                   │
│                              ┌────────────────▼──────────────┐    │
│                              │         Module 4               │    │
│                              │   AI Analysis (Gemini 2.0)    │    │
│                              │   + Offline Fallback Engine   │    │
│                              └────────────────┬──────────────┘    │
│                                               │                   │
│                                       ┌───────▼──────┐            │
│                                       │data/reports/ │            │
│                                       │ai_analysis   │            │
│                                       │.html         │            │
│                                       └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

Data flows from **left to right**, with each stage reading from the previous stage's output directory. Every module can be run independently or skipped via command-line flags.

---

## Features

### Module 1 — Data Collection
- **Stock Prices** — 1-year OHLCV history for any ticker via `yfinance`
- **Financial Statements** — Quarterly income statements, balance sheets, and cash flow statements
- **News & Sentiment** — Recent financial headlines via NewsAPI (last 7 days)
- **FX Rates** — USD/EUR, USD/JPY, USD/VND daily rates via Alpha Vantage
- **Commodity Prices** — Gold (GC=F) and WTI Crude Oil (CL=F) futures via `yfinance`
- Retry logic with exponential back-off on all API calls
- Rate-limit compliance for Alpha Vantage free tier (5 requests/min)
- JSON manifest file saved after every collection run

### Module 2 — Data Cleaning & Feature Engineering
- **Column normalisation** — handles yfinance MultiIndex columns automatically
- **Duplicate detection** — removes duplicate dates, logs count
- **Missing value imputation** — forward-fill then backward-fill (no look-ahead bias)
- **Outlier detection** — Z-score (threshold 3.5σ) on returns + IQR (multiplier 3.0) on prices; flagged, not removed
- **Feature engineering:**
  - Daily percentage returns and log returns
  - SMA-7, SMA-30 (Simple Moving Averages)
  - EMA-7, EMA-30 (Exponential Moving Averages)
  - Bollinger Bands (20-day, ±2σ)
  - RSI-14 (Relative Strength Index, Wilder smoothing)
  - Annualised rolling volatility (7-day, 30-day windows)
- Cleaning audit report saved as `data/clean/cleaning_report.json`

### Module 3 — Visualisation
- **4 required chart types** (see [Visualisations Produced](#visualisations-produced))
- **Bonus** — Interactive candlestick chart (Plotly HTML)
- Publication-quality styling: 200 DPI, consistent colour palette, labelled axes
- Agg backend — works in headless/server environments

### Module 4 — AI Analysis
- **Primary:** Google Gemini 2.0 Flash via Google AI Studio API
- **Fallback:** Built-in offline analysis engine (runs without any API key)
- Structured data context (JSON, 17 numeric fields) injected into every prompt
- Chart images sent alongside prompts for multimodal analysis
- **Four analysis outputs per run:**
  - Trend summary per asset
  - Anomaly & notable event detection
  - Risk commentary with risk rating
  - Cross-asset comparison with Sharpe proxy
- Full HTML report with embedded charts saved to `data/reports/ai_analysis.html`

---

## Project Structure

```
FinAgentProjectOggy-main/
│
├── main.py                        # Pipeline orchestrator (CLI entry point)
├── config.py                      # Shared configuration & environment loading
├── requirements.txt               # All Python dependencies with version pins
├── env.example                    # API key template — copy to .env before use
├── .gitignore                     # Excludes .env, __pycache__, data/, etc.
├── FinAgentOggy.ipynb             # Jupyter Notebook — interactive walkthrough
│
├── Modules/
│   ├── module1_data_collection.py  # Stage 1: Multi-source data collection
│   ├── module2_data_cleaning.py    # Stage 2: Cleaning & feature engineering
│   ├── module3_visualization.py    # Stage 3: Chart generation (matplotlib/plotly)
│   └── module4_ai_analysis.py      # Stage 4: Gemini LLM analysis & HTML report
│
└── data/                          # Auto-created on first run (git-ignored)
    ├── raw/                       # Raw CSVs + manifest.json
    ├── clean/                     # Cleaned CSVs + cleaning_report.json
    ├── charts/                    # PNG charts + interactive HTML candlestick
    └── reports/                   # ai_analysis.html (final report)
```

> **Note:** The `data/` directory is excluded from Git via `.gitignore`. It is created automatically on the first run.

---

## Setup and Installation

### Prerequisites

- Python **3.10 or higher** (required for `X | Y` union type hints)
- `pip` package manager
- Git

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd FinAgentProjectOggy-main
```

### Step 2 — (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure API keys

```bash
cp env.example .env
```

Open `.env` and fill in your keys (see [Configuration](#configuration--api-keys) below).

---

## Configuration — API Keys

All credentials are loaded from a `.env` file using `python-dotenv`. **Never commit your `.env` file to Git** — it is already listed in `.gitignore`.

| Variable | Module | Required? | Free Tier | Where to Get |
|---|---|---|---|---|
| `GOOGLE_API_KEY` | Module 4 (AI) | Recommended | Yes | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `NEWS_API_KEY` | Module 1 (News) | Optional | Yes (100 req/day) | [newsapi.org/register](https://newsapi.org/register) |
| `ALPHA_VANTAGE_KEY` | Module 1 (FX) | Optional | Yes (5 req/min) | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |

Your `.env` file should look like this:

```dotenv
GOOGLE_API_KEY=AIzaSy...your_key_here
NEWS_API_KEY=abc123...your_key_here
ALPHA_VANTAGE_KEY=XYZ789...your_key_here
```

> **What happens without keys?**
> - Without `GOOGLE_API_KEY` → Module 4 switches to the **offline analysis engine** automatically. All four analysis types (trend, anomaly, risk, comparison) are still generated using real data.
> - Without `NEWS_API_KEY` → News collection is skipped; all other modules run normally.
> - Without `ALPHA_VANTAGE_KEY` → FX rate collection is skipped; stock and commodity data still collected.

---

## Usage

### Run the full pipeline (recommended)

```bash
python3 main.py
```

This runs all four modules in sequence and opens the final HTML report.

### Run with custom tickers

```bash
python3 main.py --tickers AAPL TSLA AMZN NVDA
```

Any ticker supported by Yahoo Finance can be used.

### Skip individual stages

```bash
# Raw data already collected — start from cleaning
python3 main.py --skip-collection

# Jump straight to AI analysis (modules 1–3 already complete)
python3 main.py --skip-collection --skip-cleaning --skip-viz

# Generate charts only, skip AI
python3 main.py --skip-collection --skip-cleaning --skip-ai
```

### Run a single module directly

Each module can be run standalone from the project root directory:

```bash
python3 Modules/module1_data_collection.py
python3 Modules/module2_data_cleaning.py
python3 Modules/module3_visualization.py
python3 Modules/module4_ai_analysis.py
```

### Interactive exploration (Jupyter Notebook)

```bash
jupyter notebook FinAgentOggy.ipynb
```

The notebook provides a step-by-step walkthrough of the full pipeline with inline outputs.

### Command-line flags reference

| Flag | Effect |
|---|---|
| `--tickers AAPL MSFT ...` | Override default ticker list |
| `--skip-collection` | Skip Module 1; use existing `data/raw/` |
| `--skip-cleaning` | Skip Module 2; use existing `data/clean/` |
| `--skip-viz` | Skip Module 3; use existing `data/charts/` |
| `--skip-ai` | Skip Module 4; stop after visualisation |

---

## Data Sources

| Source Type | API / Library | Data Collected | Frequency |
|---|---|---|---|
| Stock Prices | `yfinance` | OHLCV daily data, 1-year history, adjusted close | Per run |
| Financial Statements | `yfinance` | Quarterly income statement, balance sheet, cash flow | Per run |
| News & Sentiment | NewsAPI (REST) | Headlines, sources, URLs — last 7 days | Per run |
| FX Rates | Alpha Vantage (REST) | USD/EUR, USD/JPY, USD/VND daily OHLC | Per run |
| Commodity Prices | `yfinance` | Gold (GC=F) and WTI Crude Oil (CL=F) futures, 1-year | Per run |

All raw data is saved to `data/raw/` as CSV files and an accompanying `manifest.json` that records the exact collection timestamp.

---

## Data Cleaning Pipeline

Module 2 applies the following steps **in order** to each ticker independently:

1. **Load** — Read raw CSV; detect and flatten MultiIndex columns from yfinance
2. **Normalise** — Standardise column names to `Open, High, Low, Close, Adj_Close, Volume`; parse date index; coerce all values to numeric
3. **Remove duplicates** — Drop duplicate dates (keep last); log count
4. **Handle missing values** — Forward-fill → backward-fill (no look-ahead bias introduced)
5. **Detect outliers** — Flag (do not remove):
   - Return outliers: log-return Z-score > 3.5σ → column `outlier_return`
   - Price outliers: IQR method with multiplier 3.0 → column `outlier_price`
6. **Engineer features:**

| Feature | Description |
|---|---|
| `daily_return` | Percentage daily return |
| `log_return` | Log return (for volatility calculation) |
| `sma_7`, `sma_30` | 7-day and 30-day Simple Moving Averages |
| `ema_7`, `ema_30` | 7-day and 30-day Exponential Moving Averages |
| `volatility_7`, `volatility_30` | Annualised rolling volatility (×√252) |
| `bb_upper`, `bb_lower`, `bb_mid` | Bollinger Bands (20-day SMA ± 2σ) |
| `rsi_14` | RSI-14 using Wilder smoothing method |

A full cleaning audit report is saved to `data/clean/cleaning_report.json` after each run, recording duplicates removed, missing values before/after, and outlier counts per ticker.

---

## Visualisations Produced

After a full run, the following charts are saved to `data/charts/`:

### Chart 1 — Price Trend & Volume Overlay
**File:** `chart1_price_trend_{TICKER}.png`

Two-panel chart per ticker. Upper panel: adjusted close price with SMA-7, SMA-30, and Bollinger Band shading; outlier dates marked with red ✕ markers. Lower panel: volume bar chart colour-coded green (positive return day) / red (negative return day).

### Chart 2 — Correlation Heatmap
**File:** `chart2_correlation_heatmap.png`

Lower-triangular Pearson correlation heatmap of **daily log returns** across all tracked assets. Computed on log returns (not prices) to avoid spurious non-stationarity correlation. RdYlGn diverging colour map; annotated with exact coefficients.

### Chart 3 — Return Distribution
**File:** `chart3_return_distribution.png`

Per-ticker subplot grid showing: histogram (60 bins) + KDE overlay + fitted Gaussian distribution. Shaded ±1σ and ±2σ bands. Daily returns expressed in percentage terms.

### Chart 4 — Rolling Statistics Dashboard
**File:** `chart4_rolling_stats_{TICKER}.png`

Three-panel chart per ticker:
- Panel 1: Price + EMA-7 + EMA-30 + Bollinger Bands
- Panel 2: Annualised 30-day rolling volatility (filled area) with mean reference line
- Panel 3: RSI-14 with overbought (70) and oversold (30) threshold bands

### Bonus — Interactive Candlestick Chart
**File:** `chart_bonus_candlestick_{TICKER}.html`

Plotly OHLC candlestick with volume subplot for the last 90 trading days. Fully interactive (zoom, pan, hover). Open directly in any browser.

---

## AI Analysis Module

Module 4 sends **structured data context + chart images** to Google Gemini and generates four types of analysis:

### Analysis Types

| Analysis | Content | Word Target |
|---|---|---|
| **Trend Summary** | Price direction, SMA signals, Bollinger Band width, key moves | 150–200 words |
| **Anomaly Detection** | Volatility spikes, RSI extremes, outlier events, likely catalysts | 150–200 words |
| **Risk Commentary** | Risk rating (Low/Moderate/High/Very High), leptokurtosis, investor profile | 150–200 words |
| **Cross-Asset Comparison** | YTD ranking, volatility ranking, Sharpe proxy, portfolio insight | 200–250 words |

### Prompt Engineering

Each LLM call receives:
1. The relevant chart image (PNG, sent as multimodal input)
2. A **structured JSON data context** with 17 fields: latest close, 52-week high/low, YTD return, annualised volatility, 30-day volatility, RSI-14, SMA values, outlier count and dates, max drawdown, and best single-day return
3. A task-specific prompt with explicit word count, numbered sub-tasks, and a prohibition on fabricating data points
4. A system instruction defining the analyst persona and requiring numbered, data-referenced output

Temperature is set to **0.3** to favour factual, deterministic outputs over creative variance.

### Offline Fallback Engine

If the Gemini API is unavailable (no key, rate limit, no network), Module 4 **automatically switches** to a built-in rule-based analysis engine that:
- Classifies trend direction from YTD return into five categories
- Describes price-vs-SMA relationships in conditional prose
- Positions the stock within its 52-week range as a percentile
- Computes a fat-tail ratio (`worst_day / estimated_daily_σ`) for leptokurtosis assessment
- Ranks assets by Sharpe proxy (YTD return / annualised volatility) for the comparison

The offline engine produces **full-length, data-grounded analysis** indistinguishable in structure from the Gemini output.

### Output

The final HTML report (`data/reports/ai_analysis.html`) embeds:
- All chart images (base64 encoded — **no external dependencies**)
- All four AI analysis sections per ticker
- Colour-coded commentary blocks (blue for trend/anomaly, amber for risk)
- The cross-asset comparison with correlation heatmap

Open the HTML file in any browser. It is fully self-contained.

---

## Output Files Reference

| File | Location | Description |
|---|---|---|
| `prices_{TICKER}.csv` | `data/raw/` | Raw OHLCV price data |
| `stmt_{TICKER}_{type}.csv` | `data/raw/` | Financial statements (income/balance/cashflow) |
| `news_headlines.csv` | `data/raw/` | Recent news headlines |
| `fx_{FROM}_{TO}.csv` | `data/raw/` | FX rate data |
| `commodity_{name}.csv` | `data/raw/` | Gold and oil price data |
| `manifest.json` | `data/raw/` | Collection timestamp and summary |
| `clean_{TICKER}.csv` | `data/clean/` | Cleaned data with all engineered features |
| `cleaning_report.json` | `data/clean/` | Per-ticker cleaning audit log |
| `chart1_price_trend_{TICKER}.png` | `data/charts/` | Price trend + volume chart |
| `chart2_correlation_heatmap.png` | `data/charts/` | Return correlation heatmap |
| `chart3_return_distribution.png` | `data/charts/` | Return distribution plots |
| `chart4_rolling_stats_{TICKER}.png` | `data/charts/` | Rolling statistics dashboard |
| `chart_bonus_candlestick_{TICKER}.html` | `data/charts/` | Interactive candlestick |
| `ai_analysis.html` | `data/reports/` | **Final report — open this in browser** |
| `ai_analyses_raw.json` | `data/reports/` | Raw JSON of all AI-generated text |
| `collection.log` | `data/` | Collection module runtime log |
| `cleaning.log` | `data/` | Cleaning module runtime log |
| `ai_analysis.log` | `data/` | AI analysis module runtime log |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'yfinance'`
Make sure your virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `Empty response for {TICKER}` in logs
Yahoo Finance occasionally returns empty data for short time periods or delisted tickers. The retry wrapper will attempt 3 times before logging an error and continuing. Verify the ticker is valid at [finance.yahoo.com](https://finance.yahoo.com).

### Alpha Vantage returns `"Information": "Thank you for using Alpha Vantage..."`
This means the **rate limit** (5 requests/minute) was hit. The 12-second inter-request sleep is already implemented; if you see this, wait 60 seconds and re-run with `--skip-collection` replaced by running only Module 1.

### `[Gemini] All attempts failed → switching to offline analysis`
This is **expected behaviour**, not an error. The pipeline continues with the offline engine. To use Gemini, ensure `GOOGLE_API_KEY` is correctly set in your `.env` file and that your key has not exceeded the free-tier quota.

### `No clean data found. Run module2 first.`
Module 3 or 4 was run before Module 2. Run the full pipeline with `python3 main.py` or run `python3 Modules/module2_data_cleaning.py` first.

### Charts not showing expected features (e.g., missing Bollinger Bands)
This occurs if Module 2 was skipped and the clean data CSV does not contain engineered feature columns. Re-run Module 2: `python3 Modules/module2_data_cleaning.py`.

### `Python 3.9` or lower — syntax errors on `X | None` type hints
Upgrade to Python 3.10+. The codebase uses the modern union syntax enabled from Python 3.10.

---

## Dependencies

See `requirements.txt` for pinned versions. Summary by function:

| Category | Library | Version |
|---|---|---|
| Data collection | `yfinance` | ≥ 0.2.40 |
| Data collection | `requests`, `httpx` | ≥ 2.31, ≥ 0.26 |
| Data collection | `python-dotenv` | ≥ 1.0 |
| Data collection | `schedule` | ≥ 1.2 |
| Data processing | `pandas` | ≥ 2.2 |
| Data processing | `numpy` | ≥ 1.26 |
| Data processing | `scipy` | ≥ 1.12 |
| Visualisation | `matplotlib` | ≥ 3.8 |
| Visualisation | `seaborn` | ≥ 0.13 |
| Visualisation | `plotly` | ≥ 5.20 |
| AI analysis | `google-generativeai` | ≥ 0.5 |
| AI analysis | `Pillow` | ≥ 10.3 |
| Utilities | `tqdm` | ≥ 4.66 |

---

## Notes and Limitations

- **Scope:** The pipeline currently tracks equity and commodity prices only. Bond yields, options data, and alternative data sources are not included.
- **News sentiment:** News headlines are collected but not yet scored with a sentiment model (e.g., FinBERT). The AI module does not currently incorporate news data into its prompts — this is a planned extension.
- **Market hours:** Data collection does not differentiate between market-open and after-hours data; all data is sourced as daily OHLCV bars, which avoids this issue.
- **Historical only:** The pipeline is designed for historical analysis. Real-time streaming is not supported in the current version, though the `schedule` library is included for scheduled batch refresh.
- **Correlation caveat:** All four default tickers (AAPL, MSFT, GOOGL, NVDA) belong to the US large-cap technology sector. Correlation between them is structurally high; the heatmap should be interpreted in that context.
- **Not financial advice:** All outputs, including AI-generated commentary, are for **educational purposes only** and do not constitute investment advice.

---

*FinAgent — IT Application in Banking and Finance, 2026 · Educational use only*
