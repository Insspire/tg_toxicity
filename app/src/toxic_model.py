import json
import os
from typing import List, Dict, Any, Optional, Callable

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "app/models/toxicity-multilabel-hn"
THRESHOLD = 0.7
BATCH_SIZE = 32
MAX_LENGTH = 512

LABELS = ["NORMAL", "INSULT", "THREAT", "OBSCENITY"]

LABELS_RU = ["Нейтральный", "Оскорбление/мат", "Угроза", "Непристойность"]
LABEL_DESCRIPTIONS_RU = {
    "Нейтральный": "нейтральные комментарии пользователей",
    "Оскорбление/мат": "комментарии, унижающие человека",
    "Угроза": "комментарии с явным намерением причинить вред другому человеку",
    "Непристойность": "комментарии с описанием или угрозой сексуального насилия",
}

TOXIC_CATEGORIES_RU = ["Оскорбление/мат", "Угроза", "Непристойность"]

class ToxicityModel:
    def __init__(self, model_path: str = MODEL_PATH, threshold: float = THRESHOLD, batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Загружаем конфигурацию меток, если она есть
        label_config_path = os.path.join(model_path, "label_config.json")
        if os.path.exists(label_config_path):
            with open(label_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.labels = config.get("labels", LABELS)
                self.labels_ru = config.get("labels_ru", LABELS_RU)
        else:
            self.labels = LABELS
            self.labels_ru = LABELS_RU

    @torch.no_grad()
    def predict_proba(self, texts: List[str], progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, float]]:
        """
        Возвращает вероятности для каждого класса для каждого текста
        Обрабатывает тексты батчами для эффективной работы с большими объемами данных
        
        Args:
            texts: Список текстов для анализа
            progress_callback: Опциональная функция для отслеживания прогресса (current, total)
        """
        all_results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        # Обрабатываем тексты батчами
        for batch_idx, i in enumerate(range(0, len(texts), self.batch_size), 1):
            batch_texts = texts[i:i + self.batch_size]
            
            # Токенизация батча
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            
            # Предсказание
            outputs = self.model(**enc)
            
            # Применяем сигмоиду для получения вероятностей
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Формируем результаты для батча
            for prob in probs:
                result = {}
                for idx, label_ru in enumerate(self.labels_ru):
                    result[label_ru] = float(prob[idx])
                all_results.append(result)
            
            # Обновляем прогресс
            if progress_callback:
                progress_callback(batch_idx, total_batches)
        
        return all_results

    def predict(self, texts: List[str], progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """
        Предсказывает метки для текстов на основе порога
        
        Args:
            texts: Список текстов для анализа
            progress_callback: Опциональная функция для отслеживания прогресса (current, total)
        """
        probs = self.predict_proba(texts, progress_callback=progress_callback)
        results = []
        
        for prob_dict in probs:
            # Определяем активные метки (выше порога)
            active_labels = []
            max_prob = 0.0
            max_label = "Нейтральный"
            
            for label_ru in self.labels_ru:
                prob_value = prob_dict[label_ru]
                if prob_value > self.threshold:
                    active_labels.append(label_ru)
                if prob_value > max_prob:
                    max_prob = prob_value
                    max_label = label_ru
            
            if not active_labels:
                active_labels = ["Нейтральный"]
            
            is_toxic = any(label in active_labels for label in TOXIC_CATEGORIES_RU)
            
            result = {
                "label": "toxic" if is_toxic else "non-toxic",
                "categories": active_labels,
                "probabilities": prob_dict,
                "max_probability": max_prob,
            }
            results.append(result)
        
        return results

    def predict_detailed(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Возвращает детальную информацию о классификации с описаниями категорий
        """
        predictions = self.predict(texts)
        results = []
        
        for pred in predictions:
            detailed = {
                "is_toxic": pred["label"] == "toxic",
                "categories": [],
                "probabilities": pred["probabilities"],
            }
            
            for category in pred["categories"]:
                detailed["categories"].append({
                    "name": category,
                    "description": LABEL_DESCRIPTIONS_RU.get(category, ""),
                    "probability": pred["probabilities"][category],
                })
            
            results.append(detailed)
        
        return results


def load_toxicity_model() -> ToxicityModel:
    """
    Удобная обертка для Streamlit: загружает модель один раз.
    """
    return ToxicityModel()
