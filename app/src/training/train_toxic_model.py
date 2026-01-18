import os
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATA_PATH = "app/data/labeled.csv"
BASE_MODEL = "cointegrated/rubert-tiny2"
OUTPUT_DIR = "models/rubert-tiny2-toxicity-custom"

@dataclass
class ModelConfig:
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


def load_data(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df.rename(columns={"comment": "text", "toxic": "label"})
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    return Dataset.from_pandas(df[["text", "label"]])


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def main():
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

    config = ModelConfig()

    print("Loading data...")
    dataset = load_data(DATA_PATH)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        problem_type="regression",
    )

    print("Tokenizing...")
    tokenized_train = dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        batched=True,
    )
    tokenized_eval = dataset["test"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        batched=True,
    )

    tokenized_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    tokenized_eval.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model=None,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
    )

    def compute_metrics(eval_pred) -> Optional[Dict[str, Any]]:
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        mse = mean_squared_error(labels, preds)
        return {"mse": mse}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Saving model to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()


