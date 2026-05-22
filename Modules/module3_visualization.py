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
