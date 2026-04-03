# model/train_bert.py

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
from torch.nn import CrossEntropyLoss

# =========================
# LOAD MAIN DATASET
# =========================
df = pd.read_csv("../data/processed/final_dataset.csv")

# =========================
# LOAD CUSTOM DATA
# =========================
custom_path = "../data/processed/custom_data.csv"

if os.path.exists(custom_path):
    custom_df = pd.read_csv(custom_path)
    df = pd.concat([df, custom_df], ignore_index=True)

# =========================
# REMOVE DUPLICATES
# =========================
df = df.drop_duplicates(subset=["text"])

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = text.replace("i'm", "i am")
    text = text.replace("can't", "cannot")
    text = text.replace("n't", " not")
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# =========================
# LABEL REFINEMENT
# =========================
def refine_labels(text, label):
    text = text.lower()

    if "frustrated" in text or "annoyed" in text:
        return "anger"
    if "anxious" in text or "worried" in text:
        return "fear"
    if "empty" in text or "lonely" in text:
        return "sadness"

    return label

df["label"] = df.apply(lambda x: refine_labels(x["text"], x["label"]), axis=1)

# =========================
# BALANCE DATASET
# =========================
df = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 2500), random_state=42)
).reset_index(drop=True)

# =========================
# SHUFFLE DATA
# =========================
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save encoder + labels
os.makedirs("saved_models", exist_ok=True)
joblib.dump(le, "saved_models/label_encoder.pkl")
joblib.dump(le.classes_, "saved_models/labels.pkl")

# =========================
# DATASET CONVERSION
# =========================
dataset = Dataset.from_pandas(df)

# Split
dataset = dataset.train_test_split(test_size=0.2)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(example):
    return tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# Rename label column
dataset = dataset.rename_column("label", "labels")

# Format dataset
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# =========================
# MODEL
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(le.classes_)
)

# =========================
# CLASS WEIGHTS
# =========================
class_counts = df['label'].value_counts().sort_index()
weights = 1.0 / class_counts
weights = torch.tensor(weights.values, dtype=torch.float)

# =========================
# CUSTOM TRAINER
# =========================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=weights.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =========================
# TRAINING ARGUMENTS
# =========================
training_args = TrainingArguments(
    output_dir="./bert_results",
    learning_rate=1.5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=4,
    logging_steps=100,
    save_strategy="no"
)

# =========================
# TRAINER
# =========================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train']
)

print("🚀 Training Improved DistilBERT...")

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE MODEL
# =========================
save_path = os.path.join(os.path.dirname(__file__), "saved_models", "bert_model")
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Improved DistilBERT model trained successfully!")