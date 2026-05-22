"""
FinAgent - Module 3: Visualization
====================================
Generates four required chart categories from cleaned data:
  1. Price trend + volume overlay (line chart with dual Y-axis)
  2. Correlation heatmap across all assets
  3. Distribution plot (histogram + KDE) of daily returns
  4. Rolling statistics chart (SMA, EMA, Bollinger Bands, RSI)

All charts are saved to data/charts/ as high-resolution PNGs
(and an optional interactive Plotly HTML version).

Setup:
    pip install matplotlib seaborn plotly pandas numpy
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

# Anchor to project root (parent of Modules/ folder) so paths are correct
# regardless of which directory you launch from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR  = _PROJECT_ROOT / "data/clean"
CHART_DIR  = _PROJECT_ROOT / "data/charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality matplotlib style
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "legend.framealpha": 0.8,
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Stores {chart_name: file_path} for the AI analysis module
CHART_REGISTRY: dict[str, Path] = {}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_clean(ticker: str) -> pd.DataFrame | None:
    path = CLEAN_DIR / f"clean_{ticker}.csv"
    if not path.exists():
        log.warning(f"[Load] {path} not found — skipping.")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_all_clean(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return {t: df for t in tickers if (df := load_clean(t)) is not None}


# ── Chart 1: Price Trend + Volume Overlay ────────────────────────────────────

def chart_price_trend(ticker: str, df: pd.DataFrame, last_n_days: int = 252) -> Path:
    """
    Two-panel chart:
      Top  : Close price with 7-day & 30-day SMA
      Bottom: Trading volume as a bar chart
    """
    df = df.tail(last_n_days).copy()
    price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    fig.suptitle(f"{ticker} — Price Trend & Volume (Last {last_n_days} Trading Days)",
                 fontsize=14, fontweight="bold")

    # Price + SMAs
    ax1.plot(df.index, df[price_col], color=PALETTE[0], lw=1.5, label="Close Price", zorder=3)
    if "sma_7" in df.columns:
        ax1.plot(df.index, df["sma_7"],  color=PALETTE[1], lw=1.2, linestyle="--", label="SMA 7d")
    if "sma_30" in df.columns:
        ax1.plot(df.index, df["sma_30"], color=PALETTE[2], lw=1.2, linestyle="-.", label="SMA 30d")

    # Bollinger Bands shading
    if "bb_upper" in df.columns:
        ax1.fill_between(df.index, df["bb_lower"], df["bb_upper"],
                         alpha=0.08, color=PALETTE[3], label="Bollinger Bands")
        ax1.plot(df.index, df["bb_upper"], color=PALETTE[3], lw=0.8, linestyle=":")
        ax1.plot(df.index, df["bb_lower"], color=PALETTE[3], lw=0.8, linestyle=":")

    # Flag outlier dates
    if "outlier_return" in df.columns:
        outliers = df[df["outlier_return"]]
        ax1.scatter(outliers.index, outliers[price_col],
                    color="red", zorder=5, s=40, label="Outlier", marker="x")

    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.2f}"))

    # Volume
    colors = [PALETTE[0] if r >= 0 else PALETTE[3]
              for r in df["daily_return"].fillna(0)]
    ax2.bar(df.index, df["Volume"].fillna(0), color=colors, alpha=0.6, width=1)
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    path = CHART_DIR / f"chart1_price_trend_{ticker}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[Chart 1] Saved → {path}")
    CHART_REGISTRY[f"price_trend_{ticker}"] = path
    return path


# ── Chart 2: Correlation Heatmap ─────────────────────────────────────────────

def chart_correlation_heatmap(data: dict[str, pd.DataFrame]) -> Path:
    """
    Correlation heatmap of daily log returns across all assets.
    """
    # Align all series on common dates
    returns = pd.DataFrame({
        ticker: df["log_return"]
        for ticker, df in data.items()
        if "log_return" in df.columns
    }).dropna()

    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(max(6, len(data)), max(5, len(data))))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle mask

    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt=".2f", linewidths=0.5,
        cmap="RdYlGn", vmin=-1, vmax=1,
        mask=mask,
        cbar_kws={"label": "Pearson Correlation"},
        square=True,
    )
    ax.set_title("Asset Return Correlation Heatmap", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    path = CHART_DIR / "chart2_correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[Chart 2] Saved → {path}")
    CHART_REGISTRY["correlation_heatmap"] = path
    return path


# ── Chart 3: Return Distribution (Histogram + KDE) ───────────────────────────

def chart_return_distribution(data: dict[str, pd.DataFrame]) -> Path:
    """
    Overlaid histogram + KDE of daily returns for all assets.
    Adds vertical dashed lines at ±1σ and ±2σ for each ticker.
    """
    n = len(data)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = list(np.array(axes).flatten())

    fig.suptitle("Daily Return Distributions", fontsize=14, fontweight="bold")

    for ax, (ticker, df), color in zip(axes, data.items(), PALETTE):
        ret = df["daily_return"].dropna() * 100   # convert to %

        ax.hist(ret, bins=60, color=color, alpha=0.35, density=True, label="Histogram")

        # KDE via seaborn
        sns.kdeplot(ret, ax=ax, color=color, lw=2, label="KDE")

        # Gaussian fit
        mu, sigma = ret.mean(), ret.std()
        x = np.linspace(ret.min(), ret.max(), 300)
        ax.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color="black", lw=1.5, linestyle="--", label=f"Normal fit\nμ={mu:.2f}%, σ={sigma:.2f}%")

        # ±1σ, ±2σ shading
        for k, alpha in [(1, 0.15), (2, 0.07)]:
            ax.axvspan(mu - k * sigma, mu + k * sigma, color=color, alpha=alpha)

        ax.axvline(0, color="grey", lw=1, linestyle=":")
        ax.set_title(f"{ticker}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = CHART_DIR / "chart3_return_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[Chart 3] Saved → {path}")
    CHART_REGISTRY["return_distribution"] = path
    return path


# ── Chart 4: Rolling Statistics (MA + Bollinger + RSI) ───────────────────────

def chart_rolling_statistics(ticker: str, df: pd.DataFrame, last_n_days: int = 180) -> Path:
    """
    Three-panel chart for one ticker:
      Panel 1: Price + EMA 7 + EMA 30 + Bollinger Bands
      Panel 2: Annualised rolling 30-day volatility
      Panel 3: RSI (14-day) with overbought/oversold bands
    """
    df = df.tail(last_n_days).copy()
    price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 1.2], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    fig.suptitle(f"{ticker} — Rolling Statistics (Last {last_n_days} Trading Days)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Price + EMAs + Bollinger
    ax1.plot(df.index, df[price_col], color=PALETTE[0], lw=1.8, label="Close")
    if "ema_7" in df.columns:
        ax1.plot(df.index, df["ema_7"],  color=PALETTE[1], lw=1.2, linestyle="--", label="EMA 7d")
    if "ema_30" in df.columns:
        ax1.plot(df.index, df["ema_30"], color=PALETTE[2], lw=1.2, linestyle="-.", label="EMA 30d")
    if "bb_upper" in df.columns:
        ax1.fill_between(df.index, df["bb_lower"], df["bb_upper"],
                         alpha=0.10, color=PALETTE[3])
        ax1.plot(df.index, df["bb_upper"], color=PALETTE[3], lw=0.8, linestyle=":")
        ax1.plot(df.index, df["bb_lower"], color=PALETTE[3], lw=0.8, linestyle=":", label="Bollinger Bands")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.2f}"))
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Panel 2: Volatility
    if "volatility_30" in df.columns:
        ax2.fill_between(df.index, df["volatility_30"] * 100, alpha=0.4,
                         color=PALETTE[4], label="30d Volatility (ann.)")
        ax2.plot(df.index, df["volatility_30"] * 100, color=PALETTE[4], lw=1.2)
        ax2.axhline(df["volatility_30"].mean() * 100, color="grey",
                    lw=1, linestyle="--", label="Mean vol.")
    ax2.set_ylabel("Volatility (%)")
    ax2.legend(loc="upper left", fontsize=8)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Panel 3: RSI
    if "rsi_14" in df.columns:
        ax3.plot(df.index, df["rsi_14"], color=PALETTE[5], lw=1.5, label="RSI 14d")
        ax3.axhline(70, color="red",   lw=1, linestyle="--", alpha=0.7, label="Overbought (70)")
        ax3.axhline(30, color="green", lw=1, linestyle="--", alpha=0.7, label="Oversold (30)")
        ax3.fill_between(df.index, df["rsi_14"], 70,
                         where=(df["rsi_14"] > 70), alpha=0.15, color="red")
        ax3.fill_between(df.index, df["rsi_14"], 30,
                         where=(df["rsi_14"] < 30), alpha=0.15, color="green")
        ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    path = CHART_DIR / f"chart4_rolling_stats_{ticker}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[Chart 4] Saved → {path}")
    CHART_REGISTRY[f"rolling_stats_{ticker}"] = path
    return path


# ── Bonus: Interactive Plotly Candlestick ─────────────────────────────────────

def chart_candlestick_plotly(ticker: str, df: pd.DataFrame, last_n_days: int = 90) -> Path:
    """
    Interactive OHLC candlestick with volume bars — saved as standalone HTML.
    """
    df = df.tail(last_n_days).copy()
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        log.warning(f"[Candlestick] {ticker}: missing OHLC columns — skipping.")
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
    ), row=1, col=1)

    if "Volume" in df.columns:
        colors = ["green" if r >= 0 else "red"
                  for r in df["daily_return"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"].fillna(0),
            marker_color=colors, opacity=0.5, name="Volume",
        ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Candlestick (Last {last_n_days} Days)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
    )
    path = CHART_DIR / f"chart_bonus_candlestick_{ticker}.html"
    fig.write_html(str(path))
    log.info(f"[Candlestick] Saved → {path}")
    CHART_REGISTRY[f"candlestick_{ticker}"] = path
    return path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_visualization(tickers: list[str] | None = None) -> dict[str, Path]:
    """
    Generate all charts for all tickers found in data/clean/.
    Returns CHART_REGISTRY mapping chart name → file path.
    """
    log.info("=" * 60)
    log.info("FinAgent Visualization — started")
    log.info("=" * 60)

    if tickers is None:
        tickers = sorted({
            p.stem.replace("clean_", "")
            for p in CLEAN_DIR.glob("clean_*.csv")
        })
        log.info(f"Auto-discovered tickers: {tickers}")

    data = load_all_clean(tickers)
    if not data:
        log.error("No clean data found. Run module2 first.")
        return {}

    # Chart 1: Price trend per ticker
    for ticker, df in data.items():
        chart_price_trend(ticker, df)

    # Chart 2: Correlation (all tickers together)
    if len(data) > 1:
        chart_correlation_heatmap(data)

    # Chart 3: Return distributions
    chart_return_distribution(data)

    # Chart 4: Rolling stats per ticker
    for ticker, df in data.items():
        chart_rolling_statistics(ticker, df)

    # Bonus candlestick (interactive)
    for ticker, df in data.items():
        chart_candlestick_plotly(ticker, df)

    log.info("=" * 60)
    log.info(f"Visualization complete. Charts saved to: {CHART_DIR}")
    log.info("=" * 60)

    return dict(CHART_REGISTRY)


if __name__ == "__main__":
    registry = run_visualization()
    for name, path in registry.items():
        print(f"  {name}: {path}")
