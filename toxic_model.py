from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "models/rubert-tiny2-toxicity-custom"
THRESHOLD = 0.7  # порог токсичности (0–1)


class ToxicityModel:
    def __init__(self, model_path: str = MODEL_PATH, threshold: float = THRESHOLD):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def predict_scores(self, texts: List[str]) -> List[float]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)
        # Модель регрессионная: на выходе одно значение (логит) на пример
        scores = outputs.logits.squeeze(-1).cpu().numpy().tolist()
        if isinstance(scores, float):
            scores = [scores]
        # Ограничим значения 0–1 на всякий случай
        scores = [max(0.0, min(1.0, float(s))) for s in scores]
        return scores

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        scores = self.predict_scores(texts)
        result = []
        for s in scores:
            label = "toxic" if s >= self.threshold else "non-toxic"
            result.append({"label": label, "score": s})
        return result


def load_toxicity_model() -> ToxicityModel:
    """
    Удобная обертка для Streamlit: загружает модель один раз.
    """
    return ToxicityModel()


