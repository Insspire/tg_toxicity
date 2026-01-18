import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
import torch
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss


DATA_PATH = "app/data/dataset.txt"
HARD_NEGATIVES_PATH = "app/data/Inappapropriate_messages.csv"
# Базовая модель: если есть обученная multilabel — дообучаем её, иначе — toxicity
MULTILABEL_MODEL = "models/rubert-tiny2-multilabel-custom"
BASE_MODEL_TOXICITY = "models/rubert-tiny2-toxicity-custom"
OUTPUT_DIR = "models/rubert-tiny2-multilabel-custom"

# Hard negatives: порог inappropriate (тексты с грубым/мат — но не оскорбления)
# и лимит примеров, чтобы не перегрузить датасет
INAPPROPRIATE_MIN = 0.6  # >= : грубый язык, но учим как NORMAL (не INSULT)
MAX_HARD_NEGATIVES = 50_000

# Метки для классификации (английские — в датасете в файле)
LABELS = ["NORMAL", "INSULT", "THREAT", "OBSCENITY"]
# Русские метки для отображения пользователям
LABELS_RU = ["Нейтральный", "Оскорбление", "Угроза", "Непристойность"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


@dataclass
class ModelConfig:
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


def parse_dataset_line(line: str) -> tuple[str, List[int]]:
    """
    Парсит строку формата: __label__LABEL1,__label__LABEL2 текст
    Возвращает (текст, список индексов меток)
    """
    line = line.strip()
    if not line:
        return "", [0]  # NORMAL по умолчанию
    
    # Ищем все метки в начале строки
    label_pattern = r"__label__([A-Z]+)"
    matches = re.findall(label_pattern, line)
    
    # Извлекаем текст (убираем все метки)
    text = re.sub(r"__label__[A-Z]+(?:,)?\s*", "", line).strip()
    
    # Создаем бинарный вектор меток
    label_vector = [0] * NUM_LABELS
    
    if matches:
        for label in matches:
            if label in LABEL_TO_ID:
                label_vector[LABEL_TO_ID[label]] = 1
    else:
        # Если меток нет, считаем NORMAL
        label_vector[LABEL_TO_ID["NORMAL"]] = 1
    
    return text, label_vector


def load_hard_negatives(
    path: str,
    min_inappropriate: float = INAPPROPRIATE_MIN,
    max_samples: int = MAX_HARD_NEGATIVES,
    exclude_texts: Optional[set] = None,
) -> Optional[Dataset]:
    """
    Загружает hard negatives из Inappapropriate_messages.csv.
    Тексты с высоким inappropriate (грубый язык, мат) помечаются как NORMAL [1,0,0,0],
    чтобы модель не считала любой мат оскорблением (могло быть восхищение, экспрессия и т.п.).
    """
    if not os.path.exists(path):
        print(f"Hard negatives: файл {path} не найден, пропуск.")
        return None

    exclude_texts = exclude_texts or set()
    df = pd.read_csv(path, encoding="utf-8")
    if "inappropriate" not in df.columns or "text" not in df.columns:
        print(f"Hard negatives: в {path} нужны колонки 'text' и 'inappropriate', пропуск.")
        return None
    df = df.dropna(subset=["text", "inappropriate"])
    df["text"] = df["text"].astype(str).str.strip()
    df["inappropriate"] = pd.to_numeric(df["inappropriate"], errors="coerce").fillna(0)

    # только с высоким inappropriate (есть грубый/неприличный язык)
    mask = df["inappropriate"] >= min_inappropriate
    df = df[mask].drop_duplicates(subset=["text"])

    # не дублировать основной датасет
    if exclude_texts:
        df = df[~df["text"].isin(exclude_texts)]

    df = df[df["text"].str.len() > 2]
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # метка: только NORMAL (не INSULT, не THREAT, не OBSCENITY)
    label_vector = [1, 0, 0, 0]
    labels = [label_vector] * len(df)
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": labels})
    print(f"Hard negatives: загружено {len(ds)} примеров (inappropriate >= {min_inappropriate})")
    return ds


def load_data(path: str) -> Dataset:
    """
    Загружает данные из dataset.txt и преобразует в формат для обучения
    """
    texts = []
    labels = []
    
    print(f"Loading data from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} lines...")
            
            text, label_vector = parse_dataset_line(line)
            if text:  # Пропускаем пустые строки
                texts.append(text)
                labels.append(label_vector)
    
    print(f"Loaded {len(texts)} examples")
    
    # Создаем DataFrame с текстами и метками
    df = pd.DataFrame({
        "text": texts,
        "labels": labels  # Список списков [0, 1, 0, 1] для каждого примера
    })
    
    # Преобразуем в Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset


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
    main_texts = set(dataset["text"])

    # Hard negatives: грубый/неприличный язык, но не оскорбления — как NORMAL
    hn = load_hard_negatives(HARD_NEGATIVES_PATH, exclude_texts=main_texts)
    if hn is not None and len(hn) > 0:
        dataset = concatenate_datasets([dataset, hn])
        print(f"Объединённый датасет: {len(dataset)} примеров")

    # Разбиение train/validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    from transformers import AutoConfig

    # Выбор базовой модели: если есть обученная multilabel — дообучаем её
    if os.path.exists(os.path.join(MULTILABEL_MODEL, "config.json")):
        mc = AutoConfig.from_pretrained(MULTILABEL_MODEL)
        if getattr(mc, "num_labels", 0) == NUM_LABELS:
            BASE_MODEL = MULTILABEL_MODEL
            print(f"Дообучение существующей multilabel-модели: {BASE_MODEL}")
        else:
            BASE_MODEL = BASE_MODEL_TOXICITY
    else:
        BASE_MODEL = BASE_MODEL_TOXICITY

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model_config = AutoConfig.from_pretrained(BASE_MODEL)

    if model_config.num_labels == NUM_LABELS and getattr(model_config, "problem_type", "") == "multi_label_classification":
        # Уже multilabel — загружаем как есть
        print("Loading multilabel model...")
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    else:
        # Токсичность (регрессия) — создаём multilabel и копируем encoder
        print("Loading base model with original parameters...")
        base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
        model_config.num_labels = NUM_LABELS
        model_config.problem_type = "multi_label_classification"
        print("Creating new model for multi-label classification...")
        model = AutoModelForSequenceClassification.from_config(model_config)
        print("Copying encoder weights from base model...")
        model.bert.load_state_dict(base_model.bert.state_dict())
        del base_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Tokenizing...")
    tokenized_train = dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        batched=True,
    )
    tokenized_eval = dataset["test"].map(
        lambda x: tokenize_function(x, tokenizer, config.max_length),
        batched=True,
    )

    # Преобразуем labels в правильный формат для мультилейбл классификации
    # Labels должны быть списками float значений (не тензорами пока)
    def format_labels_as_float(examples):
        # Преобразуем список списков int в список списков float
        labels = examples["labels"]
        return {"labels": [[float(val) for val in label_vec] for label_vec in labels]}

    tokenized_train = tokenized_train.map(format_labels_as_float, batched=True)
    tokenized_eval = tokenized_eval.map(format_labels_as_float, batched=True)

    # Используем стандартный формат для всех полей
    # Затем в data_collator преобразуем labels в float32
    tokenized_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    tokenized_eval.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Создаем кастомный DataCollator для правильной обработки labels как float32
    def data_collator(features):
        batch = {}
        first = features[0]
        
        # Обрабатываем input_ids и attention_mask (уже тензоры после set_format)
        if "input_ids" in first:
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        if "attention_mask" in first:
            batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        
        # Labels должны быть float32 тензорами для мультилейбл классификации
        # Важно: labels должны быть в batch, иначе модель не получит их
        if "labels" in first:
            # Собираем labels из всех features
            labels_list = [f["labels"] for f in features]
            # Если это уже тензоры, stack их, иначе создаем тензор из списков
            if isinstance(labels_list[0], torch.Tensor):
                labels_tensor = torch.stack(labels_list)
            else:
                labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
            
            # Преобразуем в float32, если это не float32
            if labels_tensor.dtype != torch.float32:
                batch["labels"] = labels_tensor.float()
            else:
                batch["labels"] = labels_tensor
        else:
            # Если labels нет, это ошибка - выводим предупреждение
            print("WARNING: labels not found in features!")
            print(f"Available keys: {first.keys()}")
        
        return batch

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
        metric_for_best_model="f1",
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
    )

    def compute_metrics(eval_pred) -> Optional[Dict[str, Any]]:
        predictions, labels = eval_pred
        
        # Применяем сигмоиду для получения вероятностей
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.tensor(predictions))
        
        # Используем порог 0.5 для бинарной классификации
        preds = (probs > 0.5).int().numpy()
        labels = labels.astype(int)
        
        # Вычисляем метрики
        accuracy = accuracy_score(labels, preds)
        hamming = hamming_loss(labels, preds)
        
        # Precision, recall, F1 для каждого класса
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "hamming_loss": hamming,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Saving model to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Сохраняем информацию о метках (английские для парсинга, русские для интерфейса)
    import json
    with open(os.path.join(OUTPUT_DIR, "label_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "labels": LABELS,
            "labels_ru": LABELS_RU,
            "label_to_id": LABEL_TO_ID,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Используем GPU, если доступен
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
