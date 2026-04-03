# utils/correction.py

# Simple keyword-based override (NO probability distortion)

keyword_map = {
    "anger": ["angry", "frustrated", "frustrating", "mad", "annoyed", "irritated"],
    "sadness": ["sad", "depressed", "down", "low", "unhappy", "empty"],
    "fear": ["scared", "afraid", "fear", "nervous", "anxious", "worried"],
    "joy": ["happy", "joy", "excited", "great", "amazing", "good", "cheerful"],
    "love": ["love", "care", "affection"]
}


def keyword_override(text):
    text = text.lower()

    for emotion, keywords in keyword_map.items():
        for word in keywords:
            if word in text:
                return emotion  # override only primary

    return None