"""
evaluate_model.py

CLI for threshold‑based evaluation of regression‑style sentiment scores
against a small labeled set, and for converting continuous scores to
class labels (0=Negative, 1=Neutral, 2=Positive) for the full dataset.

Features
--------
* Optimises a symmetric threshold ±T that maximises macro F1 on a
  validation subset.
* Saves a merged CSV with `predicted_label` for every article.
* Optionally plots distribution + threshold cut‑points.

Examples
--------
# Basic evaluation and label generation
python evaluate_model.py \
       --labeled data/labeled_data.csv \
       --scored data/sentiment_results.csv \
       --out scored_with_predictions.csv \
       --merge-key title

# Use a fixed threshold (skip search)
python evaluate_model.py --labeled ... --scored ... --fixed-thresh 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate sentiment thresholds and label full dataset")
    p.add_argument("--labeled", type=Path, required=True, help="CSV with human labels (col 'label')")
    p.add_argument("--scored", type=Path, required=True, help="CSV with sentiment_score column")
    p.add_argument("--out", type=Path, default="scored_with_predictions.csv", help="Output CSV with predicted_label column")
    p.add_argument("--merge-key", default="title", help="Column to merge on (default 'title'). Use 'index' for positional merge.")
    p.add_argument("--fixed-thresh", type=float, help="Skip search and use this ±threshold directly")
    p.add_argument("--plot", action="store_true", help="Show histogram plot with thresholds")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_thresholds(scores: pd.Series | np.ndarray, neg: float, pos: float) -> list[int]:
    return [0 if s <= neg else 2 if s >= pos else 1 for s in scores]


def search_threshold(scores: pd.Series, labels: pd.Series, grid: np.ndarray) -> float:
    best_f1, best_t = 0.0, grid[0]
    for t in grid:
        preds = apply_thresholds(scores, -t, t)
        f1 = f1_score(labels, preds, average="macro")
        print(f"±{t:.2f}: Macro F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load data
    labeled_df = pd.read_csv(args.labeled)
    scored_df = pd.read_csv(args.scored)

    if args.merge_key == "index":
        labeled_df = labeled_df.reset_index().rename(columns={"index": "_idx"})
        scored_df = scored_df.reset_index().rename(columns={"index": "_idx"})
        key = "_idx"
    else:
        key = args.merge_key

    if key not in labeled_df.columns or key not in scored_df.columns:
        sys.exit(f"[error] merge key '{key}' not found in both CSVs")

    merged = pd.merge(scored_df, labeled_df[[key, "label"]], on=key, how="inner")
    if merged.empty:
        sys.exit("[error] Merge produced 0 rows; check --merge-key column")

    # Threshold determination
    if args.fixed_thresh is not None:
        best_thresh = args.fixed_thresh
        print(f"Using fixed threshold ±{best_thresh:.2f}")
    else:
        grid = np.arange(0.05, 0.5, 0.05)
        best_thresh = search_threshold(merged["sentiment_score"], merged["label"], grid)
        best_f1 = f1_score(merged["label"], apply_thresholds(merged["sentiment_score"], -best_thresh, best_thresh), average="macro")
        print(f"\n✅ Best Threshold: ±{best_thresh:.2f} → Macro F1 = {best_f1:.4f}")

    # Final evaluation
    preds = apply_thresholds(merged["sentiment_score"], -best_thresh, best_thresh)
    print("\nDetailed classification report:")
    print(classification_report(merged["label"], preds, target_names=LABEL_NAMES))

    # Apply to full scored dataset & save
    scored_df["predicted_label"] = apply_thresholds(scored_df["sentiment_score"], -best_thresh, best_thresh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(args.out, index=False)
    print(f"\n✅ Saved predicted labels to {args.out.resolve()}")

    # Optional plot
    if args.plot:
        scored_df["sentiment_score"].hist(bins=50)
        plt.title("Distribution of Sentiment Scores")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Article Count")
        plt.axvline(-best_thresh, color="red", linestyle="--", label="Negative Threshold")
        plt.axvline(best_thresh, color="green", linestyle="--", label="Positive Threshold")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
