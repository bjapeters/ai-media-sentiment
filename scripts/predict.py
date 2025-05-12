#!/usr/bin/env python3
"""
predict.py - Apply the trained (chunked) DistilBERT model to score article sentiment.

Steps:
1) Load model/tokenizer from `args.model_path`.
2) Read `args.input_csv` -> Pandas DataFrame.
3) Keep columns like 'title', 'publication' in the original DataFrame (df_original).
4) Build a smaller DataFrame (df_for_tokenizer) containing ONLY ['original_index','text'] for chunking.
5) Use map(...) with chunk_tokenize_function() to split texts >512 tokens into multiple chunks.
6) Run inference on each chunk, storing chunk-level probabilities by original_index.
7) Aggregate chunk-level probabilities -> final sentiment scores (pos - neg).
8) Merge scores back into df_original to produce a single CSV row per article.
"""

import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

def chunk_tokenize_function(examples):
    """
    Tokenize text with chunking/overflow, matching how the model was trained.
    We'll replicate `original_index` for each chunk so we can aggregate later.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
        stride=128,
        return_attention_mask=True,
    )
    # If we have overflow/chunking, link each chunk back to its "original_index"
    if "overflow_to_sample_mapping" in tokenized:
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["original_index"] = [
            examples["original_index"][i] for i in sample_mapping
        ]
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV with 'text' column (and possibly extra metadata).")
    parser.add_argument("--model_path", type=str, default="models/distilbert_v1",
                        help="Path to trained DistilBERT model.")
    parser.add_argument("--output_csv", type=str, default="sentiment_results.csv",
                        help="Where to save final CSV with sentiment scores.")
    args = parser.parse_args()

    # 1. Load model + tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 2. Load + clean data into a DataFrame
    df_original = pd.read_csv(args.input_csv)
    # Remove any leftover "label" column, if present
    df_original = df_original.drop(columns=["label"], errors="ignore")
    # Drop empty rows
    df_original = df_original.dropna(subset=["text"])
    df_original = df_original[df_original["text"].str.strip().astype(bool)]
    # Assign each row an original index
    df_original = df_original.reset_index().rename(columns={"index": "original_index"})

    # We only want to feed "text" + "original_index" into the tokenizer
    df_for_tokenizer = df_original[["original_index", "text"]].copy()

    # 3. Create HF Dataset for chunked tokenization
    dataset = Dataset.from_pandas(df_for_tokenizer)

    # 4. Map the chunk-tokenize function (which returns multiple rows per doc if needed)
    #    Remove the raw 'text' column to avoid duplicates in arrow
    tokenized_dataset = dataset.map(
        chunk_tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing with chunking"
    )
    tokenized_dataset.set_format("torch")

    # 5. Run inference chunk-by-chunk, storing probabilities in a dict
    article_probs = {}
    print("Running inference on chunked dataset...")

    with torch.no_grad():
        for i in range(len(tokenized_dataset)):
            chunk = tokenized_dataset[i]
            input_ids = chunk["input_ids"].unsqueeze(0).to(device)       # shape [1, 512]
            attention_mask = chunk["attention_mask"].unsqueeze(0).to(device)
            doc_id = int(chunk["original_index"])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # shape [3]

            article_probs.setdefault(doc_id, []).append(probs)

    print("Inference complete. Aggregating chunk-level predictions...")

    # 6. For a 3-class model with label order [neg, neu, pos],
    #    define a "sentiment_score" = mean(pos) - mean(neg) across all chunks for that article.
    final_scores = []
    for doc_id in df_original["original_index"]:
        if doc_id in article_probs:
            chunk_probs = np.array(article_probs[doc_id])  # shape [n_chunks, 3]
            avg_pos = chunk_probs[:, 2].mean()
            avg_neg = chunk_probs[:, 0].mean()
            score = avg_pos - avg_neg
        else:
            # fallback if no chunks for that doc
            score = 0.0
        final_scores.append(score)

    df_original["sentiment_score"] = final_scores

    # 7. Save to CSV
    df_original.to_csv(args.output_csv, index=False)
    print(f"Done! Saved sentiment results to {args.output_csv}")

    # 8. Print a sample
    print("Sample results:")
    print(df_original.head(5)[["original_index", "sentiment_score"]])

if __name__ == "__main__":
    main()
