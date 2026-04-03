# utils/scoring.py

emotion_scores = {
    "joy": 2,
    "love": 2,
    "surprise": 1,
    "neutral": 0,
    "sadness": -2,
    "fear": -2,
    "anger": -3,
    "disgust": -3
}


def compute_score(primary, secondary, p1, p2):
    score = (emotion_scores[primary] * p1) + (emotion_scores[secondary] * p2)
    return round(score, 2)


def get_emotion_type(emotion):
    if emotion in ["joy", "love", "surprise"]:
        return "positive"
    elif emotion in ["anger", "sadness", "fear", "disgust"]:
        return "negative"
    else:
        return "neutral"