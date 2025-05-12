#!/usr/bin/env python3
"""
train.py - Fine-tune DistilBERT on 3-class sentiment with chunking, active learning, and class weights
"""


import argparse
import pandas as pd
import torch
import wandb
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, concatenate_datasets
from torch.nn import CrossEntropyLoss

def softmax_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
        stride=128,
        return_attention_mask=True
    )

    if "overflow_to_sample_mapping" in tokenized:
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        if "labels" in examples:
            tokenized["labels"] = [examples["labels"][i] for i in sample_mapping]
        if "original_index" in examples:
            tokenized["original_index"] = [examples["original_index"][i] for i in sample_mapping]

    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", type=str, required=True)
    parser.add_argument("--unlabeled", type=str, default=None)
    parser.add_argument("--model_output", type=str, default="models/distilbert_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--active_learning_rounds", type=int, default=0)
    args = parser.parse_args()

    wandb.init(project="cms-sentiment", config=args.__dict__)

    # Check MPS availability
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and prepare labeled data
    df_labeled = pd.read_csv(args.labeled)
    df_labeled = df_labeled.dropna(subset=["text", "label"])
    df_labeled["label"] = df_labeled["label"].astype(int)

    ds_labeled = Dataset.from_pandas(df_labeled[["text", "label"]])
    ds_labeled = ds_labeled.rename_column("label", "labels")
    ds_labeled = ds_labeled.map(tokenize_function, batched=True, remove_columns=["text"])
    ds_labeled.set_format("torch")

    # Train/val split after chunking
    full_dataset = ds_labeled.train_test_split(test_size=0.2, seed=42)
    ds_train, ds_val = full_dataset["train"], full_dataset["test"]

    # Model setup with device handling
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    ).to(device)

    # Class weights on same device as model
    class_weights = torch.tensor([50/5, 50/19, 50/26], device=model.device)
    loss_fct = CrossEntropyLoss(weight=class_weights)

    training_args = TrainingArguments(
        output_dir=args.model_output,
        run_name=f"distilbert-run-{int(time.time())}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="no"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")
        }

    # Custom trainer with device-aware loss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Move labels to model's device
            labels = inputs.get("labels").to(model.device)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.model_output)

    # Active learning with chunking
    if args.unlabeled and args.active_learning_rounds > 0:
        df_unlabeled = pd.read_csv(args.unlabeled)
        df_unlabeled = df_unlabeled.dropna(subset=["text"])

        # Store original indices once
        df_unlabeled = df_unlabeled.reset_index().rename(columns={"index": "original_index"})

        for round_idx in range(args.active_learning_rounds):
            print(f"\n=== Active Learning Round {round_idx+1} ===")

            # Refresh temporary indices while preserving original_index
            current_df = df_unlabeled.reset_index(drop=True)

            # Prepare dataset with both indices
            ds_unlabeled = Dataset.from_pandas(current_df[["text", "original_index"]])

            # Rest remains unchanged...
            ds_unlabeled = ds_unlabeled.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )
            ds_unlabeled.set_format("torch")

            # Get predictions for all chunks
            predictions = trainer.predict(ds_unlabeled)
            logits = predictions.predictions
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

            # Aggregate by original text
            original_indices = ds_unlabeled["original_index"]
            avg_probs = np.stack([
                probs[original_indices == idx].mean(axis=0)
                for idx in df_unlabeled.index
            ])

            # Calculate entropy and select uncertain samples
            df_unlabeled["entropy"] = softmax_entropy(avg_probs)
            uncertain_samples = df_unlabeled.nlargest(20, "entropy")

            # Labeling interface
            print("Most uncertain samples:")
            labeled_samples = []
            for idx, row in uncertain_samples.iterrows():
                print(f"\nText: {row['text'][:500]}...")
                while True:
                    label = input("Label (0=neg, 1=neu, 2=pos, skip=skip): ").strip().lower()
                    if label in {"0", "1", "2"}:
                        labeled_samples.append({
                            "text": row["text"],
                            "labels": int(label)
                        })
                        break
                    elif label == "skip":
                        break
                    else:
                        print("Invalid input. Try again.")

            # Add new labeled data
            if labeled_samples:
                new_data = Dataset.from_pandas(pd.DataFrame(labeled_samples))
                new_data = new_data.map(tokenize_function, batched=True, remove_columns=["text"])
                ds_train = concatenate_datasets([ds_train, new_data])
                trainer.train_dataset = ds_train
                trainer.train()
                df_unlabeled = df_unlabeled.drop(uncertain_samples.index)

        trainer.save_model(f"{args.model_output}_final")

if __name__ == "__main__":
    main()
