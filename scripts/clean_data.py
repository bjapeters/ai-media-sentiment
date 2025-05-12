"""
clean_data.py

CLI utility for cleaning raw CSV exports (e.g., from ProQuest).
It fixes stray encoding glitches and optionally applies `ftfy.fix_text` to a
specified text column.

Examples
--------
# Basic cleaning, default text column "full_text"
python clean_data.py --input data/raw/data_1.csv --output data/clean/data_clean.csv

# Custom text column name
python clean_data.py --input data/raw/articles.csv \
                     --output data/clean/articles_clean.csv \
                     --text-column body
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ftfy
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw article CSV files")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV")
    parser.add_argument("--output", type=Path, required=True, help="Path to save cleaned CSV")
    parser.add_argument(
        "--text-column",
        type=str,
        default="full_text",
        help="Name of the column that contains article body text to pass through ftfy",
    )
    return parser.parse_args()


def clean_csv(input_path: Path, text_column: str) -> pd.DataFrame:
    """Read CSV, coerce to UTF‑8, and fix text column if present."""
    # pandas handles most encoding snafus; `on_bad_lines='warn'` keeps rows.
    df = pd.read_csv(input_path, encoding="utf-8", on_bad_lines="warn")

    if text_column in df.columns:
        df[text_column] = df[text_column].astype(str).apply(ftfy.fix_text)
    else:
        print(f"[warn] text column '{text_column}' not found; skipping ftfy fix.")

    # Ensure all string columns are valid UTF‑8 by re‑encoding/decoding.
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda s: s.encode("utf-8", "replace").decode("utf-8", "replace"))
        )

    return df


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} does not exist")

    df_clean = clean_csv(args.input, args.text_column)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[info] Cleaned CSV written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
