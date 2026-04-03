# 🧠 AI Powered Mental Health Monitoring System

## 📌 Overview
This project detects user emotions from text input and tracks mental health over time using AI/ML models.

## 🚀 Features
- Emotion Detection using DistilBERT
- Primary & Secondary Emotion Prediction
- Confidence Score
- Emotion Classification (Positive/Negative/Neutral)
- Mental Health Scoring System
- FastAPI Backend

## 🧠 Model Details
- Model: DistilBERT (HuggingFace Transformers)
- Dataset: GoEmotions (Google)
- Classes: joy, love, anger, fear, sadness, surprise, disgust, neutral

## 🛠 Tech Stack
- Python
- PyTorch
- HuggingFace Transformers
- FastAPI
- Pandas, NumPy

## 📦 Installation

```bash
git clone https://github.com/Atharva23094/AI-Mental-Health-System.git
cd AI-Mental-Health-System
pip install -r requirements.txt

▶️ Run API
uvicorn api.app:app --reload