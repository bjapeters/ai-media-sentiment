"""
plot_results.py

Unified plotting utility for the AI‑media‑sentiment project. It supports three
visualisations:

  1. **Raw daily sentiment curves**          (kind = "raw")
  2. **N‑day rolling‑average curves**        (kind = "rolling", default N=14)
  3. **Sentiment distribution by publication** (kind = "pubdist")

Examples
--------
# 1) Raw daily sentiment for all five outlets
python plot_results.py --csv sentiment_results.csv --kind raw

# 2) 30‑day rolling average for NYT + Wired
python plot_results.py --csv sentiment_results.csv \
                      --kind rolling --window 30 \
                      --pubs "The New York Times" "Wired"

# 3) Stacked‑bar distribution using model predictions
python plot_results.py --csv scored_with_predictions.csv --kind pubdist

Input expectations
------------------
* For **raw / rolling** plots, the CSV needs:
    date, source, sentiment_score (numeric ‑1 … 1)
* For **pubdist** plots, the CSV needs EITHER:
    • `predicted_label` (0=neg, 1=neutral, 2=pos) OR
    • `sentiment`       (strings "Negative"/"Neutral"/"Positive")
  plus a `source` column.

Outputs a Matplotlib figure (displayed; use --save to write PNG).
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PUBS: List[str] = [
    "The New York Times",
    "The Guardian",
    "The Wall Street Journal",
    "Wired",
    "USA Today",
]
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AI sentiment results")
    parser.add_argument("--csv", type=pathlib.Path, required=True, help="Input CSV")
    parser.add_argument(
        "--kind",
        choices=["raw", "rolling", "pubdist"],
        default="raw",
        help="Plot type: raw time‑series, rolling average, or publication distribution.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=14,
        help="Window size (days) for rolling average; only used with --kind rolling.",
    )
    parser.add_argument(
        "--pubs",
        nargs="*",
        default=DEFAULT_PUBS,
        help="Publications to include (ignored for pubdist if empty).",
    )
    parser.add_argument("--save", type=pathlib.Path, help="Optional PNG output path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers for raw / rolling plots
# ---------------------------------------------------------------------------

def load_for_timeseries(csv_path: pathlib.Path, pubs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date", "source", "sentiment_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV for time‑series plots must include: {', '.join(required)} (missing {missing})"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["sentiment_score"])
    df = df[df["source"].isin(pubs)]
    df = df.sort_values(["source", "date"])

    # Aggregate duplicates per (source, date)
    return (
        df.groupby(["source", "date"], as_index=False)["sentiment_score"].mean()
    )


def add_rolling(df_daily: pd.DataFrame, window: int) -> pd.DataFrame:
    df_daily["rolling"] = (
        df_daily.groupby("source")["sentiment_score"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    )
    return df_daily


def plot_timeseries(df_daily: pd.DataFrame, pubs: List[str], kind: str):
    plt.figure(figsize=(10, 6))
    for pub in pubs:
        subset = df_daily[df_daily["source"] == pub]
        y = subset["rolling"] if kind == "rolling" else subset["sentiment_score"]
        plt.plot(subset["date"], y, label=pub)

    suffix = " (Rolling Avg)" if kind == "rolling" else " (Raw)"
    plt.title(f"Sentiment Over Time by Publication{suffix}")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.legend(title="Publication")
    plt.xticks(rotation=45)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Helper for stacked bar distribution plot
# ---------------------------------------------------------------------------

def plot_pub_distribution(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)

    if "sentiment" not in df.columns and "predicted_label" not in df.columns:
        raise ValueError(
            "CSV for pubdist must contain 'sentiment' or 'predicted_label' column"
        )

    if "sentiment" not in df.columns:
        df["sentiment"] = df["predicted_label"].map(LABEL_MAP)

    grouped = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0)
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

    pct.loc[DEFAULT_PUBS].plot(
        kind="bar", stacked=True, figsize=(10, 6), colormap="Set2",
    )
    plt.title("Sentiment Distribution by Publication")
    plt.xlabel("Publication")
    plt.ylabel("Percentage of Articles")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.kind in {"raw", "rolling"}:
        df_daily = load_for_timeseries(args.csv, args.pubs)
        if args.kind == "rolling":
            df_daily = add_rolling(df_daily, args.window)
        plot_timeseries(df_daily, args.pubs, args.kind)

    elif args.kind == "pubdist":
        plot_pub_distribution(args.csv)

    else:
        raise ValueError(f"Unknown plot kind: {args.kind}")

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Figure saved to {args.save.absolute()}")

    plt.show()


if __name__ == "__main__":
    main()
