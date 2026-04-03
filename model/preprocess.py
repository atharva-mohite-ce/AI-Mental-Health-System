import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


# 🔹 CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words or w in ['not', 'no']]

    return " ".join(words)


# 🔹 MAP TO 8 EMOTION CLASSES
def map_emotion(label):

    if label in ["joy", "amusement", "approval", "gratitude", "optimism"]:
        return "joy"

    elif label in ["love", "admiration", "caring"]:
        return "love"

    elif label in ["anger", "annoyance", "disapproval"]:
        return "anger"

    elif label in ["fear", "nervousness"]:
        return "fear"

    elif label in ["sadness", "disappointment", "grief", "remorse"]:
        return "sadness"

    elif label in ["surprise", "realization"]:
        return "surprise"

    elif label in ["disgust"]:
        return "disgust"

    elif label in ["neutral"]:
        return "neutral"

    else:
        return None


# 🔹 EXTRACT LABEL FROM ONE-HOT COLUMNS
def extract_label(row):

    emotion_cols = [
        'anger','annoyance','fear','nervousness','sadness',
        'disappointment','grief','remorse','joy','amusement',
        'approval','gratitude','love','optimism','admiration',
        'caring','surprise','realization','disgust','neutral'
    ]

    selected = []

    for col in emotion_cols:
        if row[col] == 1:
            selected.append(col)

    # Priority order
    priority_order = [
        "sadness", "fear", "anger",
        "disgust", "surprise",
        "love", "joy", "neutral"
    ]

    for p in priority_order:
        for emo in selected:
            if emo == p:
                return emo

    return None


# 🔹 MAIN FUNCTION
def preprocess_dataset(output_path):

    df = pd.read_csv("../data/raw/goemotions/emotions.csv")

    print("Columns:", df.columns)

    # Extract labels
    df['label'] = df.apply(extract_label, axis=1)

    # Map emotions
    df['label'] = df['label'].apply(map_emotion)

    # Remove unwanted
    df.dropna(subset=['label'], inplace=True)

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Remove empty rows
    df = df[df['text'].str.strip() != ""]

    df = df[['text', 'label']]

    # Shuffle (no aggressive balancing)
    df = df.sample(frac=1, random_state=42)

    df.reset_index(drop=True, inplace=True)

    # Save
    df.to_csv(output_path, index=False)

    print("✅ Preprocessing completed")
    print("Total samples:", len(df))