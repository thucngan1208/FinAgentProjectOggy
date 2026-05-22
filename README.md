# FinAgent — AI-Powered Financial Data Agent

> Midterm Project — IT Application in Banking and Finance, 2026

An end-to-end pipeline that autonomously collects financial data, cleans and processes it, generates professional visualisations, and delivers AI-powered analytical summaries using Google Gemini.

---

## Features

- **Data Collection** — Stock prices, financial statements, news headlines, FX rates, and commodity prices from multiple sources (yfinance, NewsAPI, Alpha Vantage)
- **Data Cleaning** — Missing value handling, duplicate removal, outlier detection, and feature engineering (returns, SMA, EMA, Bollinger Bands, RSI)
- **Visualisation** — Price trend charts, correlation heatmap, return distribution plots, and rolling statistics dashboards
- **AI Analysis** — Automated trend summaries, anomaly detection, risk commentary, and cross-asset comparison powered by Google Gemini (with full offline fallback)

---

## Project Structure

```
finagent/
├── main.py                      # Pipeline orchestrator
├── module1_data_collection.py   # Module 1: Data collection
├── module2_data_cleaning.py     # Module 2: Cleaning & feature engineering
├── module3_visualization.py     # Module 3: Chart generation
├── module4_ai_analysis.py       # Module 4: AI analysis & HTML report
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template (copy to .env)
├── .gitignore
└── data/
    ├── raw/                     # Raw collected data
    ├── clean/                   # Cleaned & engineered data
    ├── charts/                  # Generated chart images
    └── reports/                 # Final HTML report
```

---

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd finagent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
cp .env.example .env
```
Then open `.env` and fill in your keys:

| Variable | Required | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Yes (for AI module) | https://aistudio.google.com/app/apikey |
| `NEWS_API_KEY` | Optional | https://newsapi.org/register |
| `ALPHA_VANTAGE_KEY` | Optional | https://www.alphavantage.co/support/#api-key |

> **Note:** All keys have a free tier. Module 4 also works without a Google API key using the built-in offline analysis engine.

---

## Usage

### Run the full pipeline
```bash
python3 main.py
```

### Run with custom tickers
```bash
python3 main.py --tickers AAPL TSLA AMZN NVDA
```

### Skip steps (if data already exists)
```bash
# Skip collection, run from cleaning onwards
python3 main.py --skip-collection

# Run only AI analysis (modules 1–3 already done)
python3 main.py --skip-collection --skip-cleaning --skip-viz
```

### Run a single module directly
```bash
python3 module1_data_collection.py
python3 module2_data_cleaning.py
python3 module3_visualization.py
python3 module4_ai_analysis.py
```

---

## Output

After a full pipeline run:

| Output | Location |
|---|---|
| Raw data (CSV) | `data/raw/` |
| Cleaned data + features (CSV) | `data/clean/` |
| Charts (PNG) | `data/charts/` |
| AI analysis report (HTML) | `data/reports/ai_analysis.html` |
| Collection log | `data/collection.log` |
| Cleaning log | `data/cleaning.log` |
| AI analysis log | `data/ai_analysis.log` |

Open `data/reports/ai_analysis.html` in your browser to view the full report with all charts and AI commentary.

---

## Data Sources

| Source | Data Type | API / Library |
|---|---|---|
| Yahoo Finance | Stock prices, financial statements | `yfinance` |
| NewsAPI | Financial news headlines | REST API |
| Alpha Vantage | FX exchange rates | REST API |
| Yahoo Finance | Commodity prices (Gold, Oil) | `yfinance` |

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- `yfinance`, `requests`, `python-dotenv` — data collection
- `pandas`, `numpy`, `scipy` — data processing
- `matplotlib`, `seaborn`, `plotly` — visualisation
- `google-generativeai`, `Pillow` — AI analysis

---

## Notes

- API keys are loaded from `.env` and never hardcoded. **Never commit your `.env` file.**
- Module 4 automatically falls back to offline analysis if the Gemini API is unavailable (rate limit, no key, no network).
- Default tickers: `AAPL`, `MSFT`, `GOOGL`, `NVDA`. Override with `--tickers`.

---

*This project is for educational purposes only and does not constitute financial advice.*
