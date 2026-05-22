"""
FinAgent - Module 4: AI Analysis (Google AI Studio / Gemini + Offline Fallback)
================================================================================
Sends charts and data to Gemini for professional financial analysis.
If the API is unavailable (rate limit, invalid key, no network),
automatically falls back to OFFLINE analysis using real figures from cleaned data.

Outputs:
  - Trend summary per asset
  - Anomaly / notable event identification
  - Risk commentary (based on volatility & RSI)
  - Cross-asset comparison
  - Full HTML report (data/reports/ai_analysis.html)
    → Automatically opened in VS Code after generation

Setup:
    pip install google-generativeai pandas Pillow

Environment variable (.env):
    GOOGLE_API_KEY=your_google_ai_studio_key
    (Get free key at: https://aistudio.google.com/app/apikey)
"""

import base64
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Anchor to project root (parent of Modules/ folder) so paths are correct
# regardless of which directory you launch from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR   = _PROJECT_ROOT / "data/clean"
CHART_DIR   = _PROJECT_ROOT / "data/charts"
REPORT_DIR  = _PROJECT_ROOT / "data/reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "gemini-2.0-flash"
MAX_TOKENS  = 1500
RETRY_DELAY = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/ai_analysis.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── VS Code opener ─────────────────────────────────────────────────────────────

def open_in_vscode(path: Path) -> None:
    """Open the file in VS Code and reveal it in the Explorer sidebar."""
    import webbrowser
    resolved = path.resolve()
    opened = False
    try:
        # --reuse-window opens in editor; we also pass the folder so Explorer refreshes
        result = subprocess.run(
            ["code", "--reuse-window", str(resolved.parent), str(resolved)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("✅ [VS Code] Report opened in VS Code.")
            opened = True
        else:
            log.warning(f"[VS Code] exit code {result.returncode}: {result.stderr.strip()}")
    except FileNotFoundError:
        log.warning("[VS Code] 'code' CLI not found — falling back to browser.")
    except Exception as exc:
        log.warning(f"[VS Code] Error: {exc}")

    if not opened:
        # Always open in default browser as fallback
        url = resolved.as_uri()
        webbrowser.open(url)
        log.info(f"🌐 [Browser] Report opened in browser: {url}")


# ── Gemini client ──────────────────────────────────────────────────────────────

def get_gemini_model():
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY is not set in the .env file")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=MAX_TOKENS,
        ),
        system_instruction=(
            "You are a professional financial analyst with expertise in equity markets, "
            "technical analysis, and risk management. "
            "Reference specific numbers and dates. Use precise financial terminology. "
            "Be objective and evidence-based. Do NOT fabricate data points."
        ),
    )
    return model


# ── Data context builder ───────────────────────────────────────────────────────

def build_data_context(ticker: str, df: pd.DataFrame) -> str:
    price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
    recent = df.tail(30)
    ctx = {
        "ticker":                     ticker,
        "period_start":               str(df.index.min().date()),
        "period_end":                 str(df.index.max().date()),
        "total_trading_days":         len(df),
        "latest_close":               round(float(df[price_col].iloc[-1]), 2),
        "52w_high":                   round(float(df[price_col].max()), 2),
        "52w_low":                    round(float(df[price_col].min()), 2),
        "ytd_return_pct":             round(float(df["daily_return"].sum() * 100), 2),
        "avg_daily_return_pct":       round(float(df["daily_return"].mean() * 100), 4),
        "annualised_volatility_pct":  round(float(df["log_return"].std() * (252**0.5) * 100), 2),
        "last_30d_volatility_pct":    round(float(recent["log_return"].std() * (252**0.5) * 100), 2),
        "latest_rsi_14":              round(float(df["rsi_14"].iloc[-1]), 1) if "rsi_14" in df.columns else None,
        "latest_sma_7":               round(float(df["sma_7"].iloc[-1]), 2)  if "sma_7"  in df.columns else None,
        "latest_sma_30":              round(float(df["sma_30"].iloc[-1]), 2) if "sma_30" in df.columns else None,
        "outlier_events":             int(df["outlier_return"].sum()) if "outlier_return" in df.columns else 0,
        "outlier_dates":              (
            df[df["outlier_return"]].index.strftime("%Y-%m-%d").tolist()
            if "outlier_return" in df.columns else []
        ),
        "max_daily_drawdown_pct":     round(float(df["daily_return"].min() * 100), 2),
        "best_single_day_pct":        round(float(df["daily_return"].max() * 100), 2),
    }
    return json.dumps(ctx, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE ANALYSIS — runs entirely without the API
#  Uses real figures from cleaned data to produce professional analysis
# ══════════════════════════════════════════════════════════════════════════════

def offline_trend_summary(ticker: str, ctx: dict) -> str:
    """Offline price trend analysis based on SMA, Bollinger Bands, and returns."""
    close     = ctx["latest_close"]
    sma7      = ctx.get("latest_sma_7")
    sma30     = ctx.get("latest_sma_30")
    high_52w  = ctx["52w_high"]
    low_52w   = ctx["52w_low"]
    ytd       = ctx["ytd_return_pct"]
    vol_ann   = ctx["annualised_volatility_pct"]
    vol_30d   = ctx["last_30d_volatility_pct"]
    best_day  = ctx["best_single_day_pct"]
    worst_day = ctx["max_daily_drawdown_pct"]
    start     = ctx["period_start"]
    end       = ctx["period_end"]

    # Determine overall trend
    if ytd > 20:
        trend_desc = "strong uptrend"
        trend_word = "clearly bullish"
    elif ytd > 5:
        trend_desc = "positive uptrend"
        trend_word = "moderately bullish"
    elif ytd > -5:
        trend_desc = "sideways"
        trend_word = "neutral (sideways)"
    elif ytd > -20:
        trend_desc = "moderate decline"
        trend_word = "mildly bearish"
    else:
        trend_desc = "strong decline"
        trend_word = "clearly bearish"

    # Price vs SMA relationship
    sma_comment = ""
    if sma7 and sma30:
        if close > sma7 > sma30:
            sma_comment = (f"The current price (${close}) is trading above both the SMA-7 (${sma7}) "
                           f"and SMA-30 (${sma30}), confirming short-term bullish momentum.")
        elif close < sma7 < sma30:
            sma_comment = (f"The current price (${close}) is trading below both the SMA-7 (${sma7}) "
                           f"and SMA-30 (${sma30}), indicating short-term selling pressure.")
        elif close > sma30:
            sma_comment = (f"Price (${close}) is trading above the SMA-30 (${sma30}) "
                           f"but below the SMA-7 (${sma7}), producing a mixed signal.")
        else:
            sma_comment = (f"Price (${close}) is trading below the SMA-30 (${sma30}), "
                           f"suggesting a weak medium-term trend.")

    # Position within 52-week range
    range_52w = high_52w - low_52w
    pct_from_low = ((close - low_52w) / range_52w * 100) if range_52w > 0 else 50
    if pct_from_low > 80:
        range_comment = f"The stock is trading near its 52-week high (${high_52w}), at {pct_from_low:.0f}% of its annual range."
    elif pct_from_low < 20:
        range_comment = f"The stock is trading near its 52-week low (${low_52w}), at only {pct_from_low:.0f}% of its annual range."
    else:
        range_comment = f"The stock is in the middle of its 52-week range (${low_52w}–${high_52w}), at {pct_from_low:.0f}%."

    # Volatility / Bollinger Band comment
    if vol_30d > vol_ann * 1.2:
        bb_comment = "Recent 30-day volatility exceeds the annual average, indicating Bollinger Bands are widening — the market is in a period of elevated uncertainty."
    elif vol_30d < vol_ann * 0.8:
        bb_comment = "30-day volatility is below the annual average; Bollinger Bands are contracting — this often precedes a breakout."
    else:
        bb_comment = "30-day volatility is in line with the annual average; Bollinger Bands remain in a normal state."

    return (
        f"**Trend Analysis — {ticker} ({start} → {end})**\n\n"
        f"Over the analysis period, {ticker} recorded a cumulative return of {ytd:+.2f}%, "
        f"reflecting a {trend_word} trend. {sma_comment} "
        f"{range_comment}\n\n"
        f"{bb_comment} "
        f"The best single trading day in the period reached +{best_day:.2f}%, "
        f"while the largest single-day decline was {worst_day:.2f}%. "
        f"From a technical perspective, the {trend_desc} trend in the near term "
        f"{'needs to be confirmed by increasing volume to sustain momentum.' if ytd > 0 else 'warrants close monitoring of key support levels to avoid further breakdown.'}"
    )


def offline_anomaly_analysis(ticker: str, ctx: dict) -> str:
    """Offline anomaly and notable event analysis."""
    outlier_n     = ctx["outlier_events"]
    outlier_dates = ctx["outlier_dates"]
    rsi           = ctx.get("latest_rsi_14")
    vol_ann       = ctx["annualised_volatility_pct"]
    vol_30d       = ctx["last_30d_volatility_pct"]
    worst_day     = ctx["max_daily_drawdown_pct"]
    best_day      = ctx["best_single_day_pct"]

    # Outlier comment
    if outlier_n == 0:
        outlier_comment = "No abnormal return events were detected during the analysis period — price movements remained within normal statistical bounds."
    elif outlier_n == 1:
        date_str = f" on {outlier_dates[0]}" if outlier_dates else ""
        outlier_comment = f"One abnormal return event was detected{date_str}, exceeding the 3.5 standard-deviation threshold. This may be linked to an earnings release or a macro event."
    else:
        dates_str = ", ".join(outlier_dates[:3]) if outlier_dates else "undetermined"
        outlier_comment = (f"{outlier_n} anomalous events were detected (notable dates: {dates_str}). "
                           f"This frequency suggests {ticker} is sensitive to market news or company-specific events.")

    # RSI comment
    if rsi is None:
        rsi_comment = "The RSI-14 indicator is unavailable for this period."
    elif rsi > 75:
        rsi_comment = f"RSI-14 is currently at {rsi:.1f} — clearly in overbought territory; a short-term pullback is possible."
    elif rsi > 60:
        rsi_comment = f"RSI-14 is at {rsi:.1f} — leaning overbought; momentum is positive but caution is warranted."
    elif rsi < 25:
        rsi_comment = f"RSI-14 is at {rsi:.1f} — deeply oversold; a technical recovery opportunity may be emerging."
    elif rsi < 40:
        rsi_comment = f"RSI-14 is at {rsi:.1f} — leaning oversold; selling pressure remains present."
    else:
        rsi_comment = f"RSI-14 is at {rsi:.1f} — in neutral territory, with no overbought or oversold signal."

    # Volatility spike comment
    vol_ratio = vol_30d / vol_ann if vol_ann > 0 else 1
    if vol_ratio > 1.3:
        vol_comment = f"30-day volatility ({vol_30d:.1f}%) is significantly higher than the annual average ({vol_ann:.1f}%), reflecting a period of heightened recent instability."
    elif vol_ratio < 0.7:
        vol_comment = f"30-day volatility ({vol_30d:.1f}%) is well below the annual average ({vol_ann:.1f}%), indicating a relatively stable recent period."
    else:
        vol_comment = f"30-day volatility ({vol_30d:.1f}%) is in line with the annual average ({vol_ann:.1f}%), with no unusual volatility signal."

    return (
        f"**Anomaly & Event Analysis — {ticker}**\n\n"
        f"{outlier_comment}\n\n"
        f"{rsi_comment} "
        f"{vol_comment}\n\n"
        f"Extreme intra-period moves: worst single day {worst_day:.2f}%, "
        f"best single day +{best_day:.2f}%. "
        f"{'These anomalies appear to have been absorbed and are no longer present in recent data.' if outlier_n <= 1 else 'These anomalies indicate the stock is susceptible to external shocks and warrants close monitoring.'}"
    )


def offline_risk_commentary(ticker: str, ctx: dict) -> str:
    """Offline risk analysis based on volatility, drawdown, and RSI."""
    vol_ann   = ctx["annualised_volatility_pct"]
    vol_30d   = ctx["last_30d_volatility_pct"]
    worst_day = ctx["max_daily_drawdown_pct"]
    best_day  = ctx["best_single_day_pct"]
    rsi       = ctx.get("latest_rsi_14")
    ytd       = ctx["ytd_return_pct"]

    # Risk rating
    if vol_ann > 40:
        rating = "Very High"
        rating_desc = "annualised volatility exceeds 40%; suitable only for investors with high risk tolerance"
    elif vol_ann > 25:
        rating = "High"
        rating_desc = "annualised volatility above 25%, exceeding the typical large-cap benchmark (~15–20%)"
    elif vol_ann > 15:
        rating = "Moderate"
        rating_desc = "annualised volatility in the 15–25% range, consistent with the broader market"
    else:
        rating = "Low"
        rating_desc = "annualised volatility below 15%, below the typical market benchmark"

    # Fat tail assessment
    tail_ratio = abs(worst_day) / (vol_ann / 16)  # ~daily vol estimate
    if tail_ratio > 3:
        tail_comment = f"The worst single-day decline ({worst_day:.2f}%) is {tail_ratio:.1f}× the estimated daily standard deviation, indicating fat tails (leptokurtosis) — extreme risk is higher than a normal distribution would suggest."
    else:
        tail_comment = f"The worst single-day decline ({worst_day:.2f}%) falls within 3 standard deviations, suggesting a relatively symmetric return distribution."

    # Investor profile
    if rating in ("Very High", "High"):
        profile = "Aggressive Growth investors with a long investment horizon (5+ years) and the ability to withstand temporary drawdowns exceeding 30%."
    elif rating == "Moderate":
        profile = "Growth investors with a medium-to-long horizon (3–5 years) and a moderate risk tolerance."
    else:
        profile = "Suitable for a broad range of investors, including conservative investors seeking steady growth."

    return (
        f"**Risk Commentary — {ticker}**\n\n"
        f"🔴 Risk Rating: **{rating}** — {rating_desc}.\n\n"
        f"Annualised volatility stands at {vol_ann:.1f}%, versus the typical large-cap benchmark of ~15–20%. "
        f"30-day rolling volatility is {vol_30d:.1f}%. "
        f"{tail_comment}\n\n"
        f"The best single-day return reached +{best_day:.2f}%, highlighting the upside potential when a catalyst is present. "
        f"YTD return is {ytd:+.2f}%.\n\n"
        f"**Suitable investor profile:** {profile}"
    )


def offline_comparison(all_data: dict) -> str:
    """Offline cross-asset comparison based on return, volatility, and RSI."""
    summaries = {}
    for ticker, df in all_data.items():
        ctx = json.loads(build_data_context(ticker, df))
        summaries[ticker] = ctx

    # Sort by YTD return
    sorted_by_return = sorted(summaries.items(), key=lambda x: x[1]["ytd_return_pct"], reverse=True)
    best_ticker,  best_ctx  = sorted_by_return[0]
    worst_ticker, worst_ctx = sorted_by_return[-1]

    # Sort by volatility
    sorted_by_vol = sorted(summaries.items(), key=lambda x: x[1]["annualised_volatility_pct"])
    lowest_vol_ticker  = sorted_by_vol[0][0]
    highest_vol_ticker = sorted_by_vol[-1][0]

    # Sharpe proxy (return / volatility)
    for t, ctx in summaries.items():
        vol = ctx["annualised_volatility_pct"]
        ctx["sharpe_proxy"] = round(ctx["ytd_return_pct"] / vol, 3) if vol > 0 else 0

    best_sharpe  = max(summaries.items(), key=lambda x: x[1]["sharpe_proxy"])
    worst_sharpe = min(summaries.items(), key=lambda x: x[1]["sharpe_proxy"])

    # Performance summary table
    lines = []
    for ticker, ctx in sorted_by_return:
        lines.append(
            f"  • {ticker}: YTD {ctx['ytd_return_pct']:+.1f}%, "
            f"Vol {ctx['annualised_volatility_pct']:.1f}%, "
            f"RSI {ctx['latest_rsi_14'] or 'N/A'}, "
            f"Sharpe proxy {ctx['sharpe_proxy']:.2f}"
        )
    table = "\n".join(lines)

    return (
        f"**Cross-Asset Comparison — {', '.join(summaries.keys())}**\n\n"
        f"Performance summary:\n{table}\n\n"
        f"**Top performer:** {best_ticker} with a YTD return of {best_ctx['ytd_return_pct']:+.1f}%. "
        f"**Worst performer:** {worst_ticker} with a YTD return of {worst_ctx['ytd_return_pct']:+.1f}%.\n\n"
        f"On the risk dimension, {lowest_vol_ticker} carries the lowest volatility ({summaries[lowest_vol_ticker]['annualised_volatility_pct']:.1f}%), "
        f"offering the best defensive characteristics in the basket. "
        f"{highest_vol_ticker} has the highest volatility ({summaries[highest_vol_ticker]['annualised_volatility_pct']:.1f}%), "
        f"making it suitable for aggressive growth investors.\n\n"
        f"**Risk-adjusted return (Sharpe proxy):** {best_sharpe[0]} leads ({best_sharpe[1]['sharpe_proxy']:.2f}), "
        f"while {worst_sharpe[0]} has the lowest risk-adjusted efficiency ({worst_sharpe[1]['sharpe_proxy']:.2f}).\n\n"
        f"**Portfolio insight:** Combining {lowest_vol_ticker} (stable anchor) with "
        f"{best_sharpe[0]} (best risk/reward ratio) can balance growth and risk control. "
        f"Assets in this basket exhibit high correlation as all belong to the US technology sector — "
        f"adding non-tech holdings would provide meaningful diversification."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CALLS with automatic fallback to offline
# ══════════════════════════════════════════════════════════════════════════════

def load_chart_image(path: Path):
    if not path.exists() or path.suffix != ".png":
        log.warning(f"[Image] Chart not found: {path}")
        return None
    try:
        img = Image.open(path).convert("RGB")
        log.info(f"[Image] Loaded {path.name} ({img.size[0]}×{img.size[1]}px)")
        return img
    except Exception as exc:
        log.error(f"[Image] Failed to load {path}: {exc}")
        return None


def call_gemini(model, parts: list, retries: int = 2) -> str | None:
    """
    Call Gemini. Returns text on success, None on failure.
    (None → caller will use the offline fallback)
    """
    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(parts)
            return response.text.strip()
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                wait = RETRY_DELAY * attempt
                log.warning(f"[Gemini] Rate limit hit. Waiting {wait}s (attempt {attempt}/{retries})...")
                time.sleep(wait)
            else:
                log.warning(f"[Gemini] Error on attempt {attempt}: {exc}")
                time.sleep(2)
    log.warning("[Gemini] All attempts failed → switching to offline analysis.")
    return None


# ── Analysis functions ─────────────────────────────────────────────────────────

def analyse_price_trend(model, ticker: str, df: pd.DataFrame) -> str:
    chart_img = load_chart_image(CHART_DIR / f"chart1_price_trend_{ticker}.png")
    data_ctx  = build_data_context(ticker, df)
    ctx       = json.loads(data_ctx)

    if model:
        prompt = f"""
You are analysing the price trend chart for {ticker}.
STRUCTURED DATA CONTEXT: {data_ctx}
Write a professional trend summary (150-200 words) covering:
1. Overall price direction over the full period
2. Price vs 7-day & 30-day SMAs
3. Bollinger Band width (increasing or decreasing volatility)
4. Most significant price move (date and magnitude)
5. One forward-looking technical note
"""
        parts  = [chart_img, prompt] if chart_img else [prompt]
        log.info(f"[AI] Trend analysis for {ticker} (Gemini)...")
        result = call_gemini(model, parts)
        if result:
            time.sleep(2)
            return result

    # Offline fallback
    log.info(f"[Offline] Trend analysis for {ticker}...")
    return offline_trend_summary(ticker, ctx)


def analyse_anomalies(model, ticker: str, df: pd.DataFrame) -> str:
    chart_img = load_chart_image(CHART_DIR / f"chart4_rolling_stats_{ticker}.png")
    data_ctx  = build_data_context(ticker, df)
    ctx       = json.loads(data_ctx)

    if model:
        prompt = f"""
You are analysing rolling statistics (EMA, volatility, RSI) for {ticker}.
STRUCTURED DATA CONTEXT: {data_ctx}
Write a professional anomaly analysis (150-200 words) covering:
1. Periods of abnormally high volatility
2. RSI extremes (overbought >70, oversold <30)
3. {ctx.get('outlier_events', 0)} flagged statistical outlier events
4. Possible macro or company catalysts
5. Whether anomalies are resolved in recent data
"""
        parts  = [chart_img, prompt] if chart_img else [prompt]
        log.info(f"[AI] Anomaly analysis for {ticker} (Gemini)...")
        result = call_gemini(model, parts)
        if result:
            time.sleep(2)
            return result

    log.info(f"[Offline] Anomaly analysis for {ticker}...")
    return offline_anomaly_analysis(ticker, ctx)


def analyse_risk(model, ticker: str, df: pd.DataFrame) -> str:
    chart_img = load_chart_image(CHART_DIR / "chart3_return_distribution.png")
    data_ctx  = build_data_context(ticker, df)
    ctx       = json.loads(data_ctx)

    if model:
        prompt = f"""
You are analysing daily return distribution for {ticker}.
STRUCTURED DATA CONTEXT: {data_ctx}
Write a professional risk commentary (150-200 words) covering:
1. Tail risk — fat tails (leptokurtosis)?
2. Annualised volatility ({ctx.get('annualised_volatility_pct')}%) vs benchmark (~15-20%)
3. Worst single-day ({ctx.get('max_daily_drawdown_pct')}%) and best ({ctx.get('best_single_day_pct')}%)
4. Risk rating (Low/Moderate/High/Very High) with justification
5. Suitable investor risk profile
"""
        parts  = [chart_img, prompt] if chart_img else [prompt]
        log.info(f"[AI] Risk analysis for {ticker} (Gemini)...")
        result = call_gemini(model, parts)
        if result:
            time.sleep(2)
            return result

    log.info(f"[Offline] Risk analysis for {ticker}...")
    return offline_risk_commentary(ticker, ctx)


def analyse_comparison(model, all_data: dict) -> str:
    chart_img = load_chart_image(CHART_DIR / "chart2_correlation_heatmap.png")

    if model:
        summary = {}
        for ticker, df in all_data.items():
            ctx = json.loads(build_data_context(ticker, df))
            summary[ticker] = {
                "ytd_return_pct":     ctx["ytd_return_pct"],
                "annualised_vol_pct": ctx["annualised_volatility_pct"],
                "latest_rsi":         ctx["latest_rsi_14"],
                "52w_high":           ctx["52w_high"],
                "52w_low":            ctx["52w_low"],
            }
        prompt = f"""
Analyse the correlation heatmap for: {', '.join(all_data.keys())}
MULTI-ASSET SUMMARY: {json.dumps(summary, indent=2)}
Write a professional cross-asset comparison (200-250 words) covering:
1. Most positively correlated assets (concentration risk)
2. Assets providing genuine diversification
3. YTD performance comparison (winner/loser)
4. Risk-adjusted returns
5. Portfolio construction insight
"""
        parts  = [chart_img, prompt] if chart_img else [prompt]
        log.info("[AI] Cross-asset comparison (Gemini)...")
        result = call_gemini(model, parts)
        if result:
            time.sleep(2)
            return result

    log.info("[Offline] Cross-asset comparison...")
    return offline_comparison(all_data)