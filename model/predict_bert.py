# model/predict_bert.py

import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from utils.scoring import compute_score, get_emotion_type
from utils.correction import keyword_override


# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "bert_model")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# Labels (must match training)
labels = [
    "joy",
    "love",
    "anger",
    "fear",
    "sadness",
    "surprise",
    "disgust",
    "neutral"
]


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # ✅ FIX: Remove token_type_ids (DistilBERT doesn't support it)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Top 2 predictions
    top_indices = probs.argsort()[-2:][::-1]

    primary = labels[top_indices[0]]
    secondary = labels[top_indices[1]]

    p1 = float(probs[top_indices[0]])
    p2 = float(probs[top_indices[1]])

    # Keyword override
    override = keyword_override(text)
    if override:
        primary = override

    confidence = round(p1, 3)

    score = float(compute_score(primary, secondary, p1, p2))
    emotion_type = get_emotion_type(primary)

    return {
        "Primary Emotion": primary,
        "Secondary Emotion": secondary,
        "Confidence": confidence,
        "Score": score,
        "Emotion Type": emotion_type
    }


# CLI test
if __name__ == "__main__":
    while True:
        try:
            text = input("\nEnter text: ")
            result = predict(text)

            print("\n🔹 Prediction Result:")
            print(result)

        except KeyboardInterrupt:
            print("\nExiting...")
            break