"""Plot paper-style portfolio baseline metrics from evaluation CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["TradingAgents", "Buy&Hold", "MACD", "KDJ&RSI", "ZMR", "SMA"]
COLORS = {
    "TradingAgents": "#E76F51",
    "Buy&Hold": "#8C8C8C",
    "MACD": "#0072B2",
    "KDJ&RSI": "#009E73",
    "ZMR": "#E69F00",
    "SMA": "#56B4E9",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", help="Path to paper_style_metrics.csv")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    out_prefix = Path(args.output_prefix) if args.output_prefix else metrics_path.with_name("fig_paper_style_baselines")
    data = pd.read_csv(metrics_path)
    portfolio = data[data["ticker"] == "PORTFOLIO"].copy()
    portfolio["method"] = pd.Categorical(portfolio["method"], ORDER, ordered=True)
    portfolio = portfolio.sort_values("method")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw={"wspace": 0.32})
    colors = [COLORS.get(method, "#B0BEC5") for method in portfolio["method"].astype(str)]

    axes[0].bar(portfolio["method"].astype(str), portfolio["CR"] * 100, color=colors)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].margins(y=0.16)
    axes[0].set_title("Cumulative Return (CR)")
    axes[0].set_ylabel("%")
    axes[0].tick_params(axis="x", rotation=28)
    for i, value in enumerate(portfolio["CR"] * 100):
        va = "bottom" if value >= 0 else "top"
        offset = 0.25 if value >= 0 else -0.25
        axes[0].text(i, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=7.5)

    axes[1].bar(portfolio["method"].astype(str), portfolio["SR"], color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].margins(y=0.18)
    axes[1].set_title("Sharpe Ratio (SR)")
    axes[1].tick_params(axis="x", rotation=28)
    for i, value in enumerate(portfolio["SR"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.10 if value >= 0 else -0.10
        axes[1].text(i, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=7.5)

    fig.suptitle(
        "Paper-style Baseline Comparison on A-share Continuous Backtest",
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(top=0.82, bottom=0.25)
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    print("Wrote:", out_prefix.with_suffix(".png"))
    print("Wrote:", out_prefix.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
