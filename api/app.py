# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.predict_bert import predict


# Initialize app
app = FastAPI(
    title="Mental Health AI API",
    description="Emotion Detection + Mental Health Scoring System",
    version="1.0"
)


# Input schema
class TextInput(BaseModel):
    text: str


# Root route
@app.get("/")
def home():
    return {
        "message": "Mental Health AI API is running 🚀"
    }


# 🔥 MAIN ENDPOINT
@app.post("/predict")
def get_prediction(data: TextInput):
    result = predict(data.text)
    return result