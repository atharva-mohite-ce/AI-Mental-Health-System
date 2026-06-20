# utils/correction.py

keyword_map = {
    "anger": [
        "angry", "frustrated", "frustrating", "mad", "annoyed", "irritated",
        "furious", "rage", "outraged", "hostile", "aggressive", "livid",
        "pissed", "hate", "hating", "hatred", "bitter", "yelled", "shouted", "conflict"
    ],
    "sadness": [
        "sad", "depressed", "depression", "down", "low", "unhappy", "empty",
        "lonely", "hopeless", "helpless", "miserable", "heartbroken",
        "grief", "grieving", "hurt", "pain", "crying", "tears", "dying",
        "dead inside", "numb", "lost", "broken", "suffering", "devastated",
        "not feeling well", "not well", "terrible", "awful", "horrible",
        "worst", "bad day", "really bad", "so bad", "feel like dying",
        "feel like i am dying", "feel like im dying", "miss", "missing",
        "disappointed", "disappointment", "failure", "failed", "exhausted",
        "drained", "tired of", "cant take", "cannot take", "hurt me", "hurts me", "dont know what", "don't know what", "lost myself", "not myself"
    ],
    "fear": [
        "scared", "afraid", "fear", "nervous", "anxious", "worried",
        "anxiety", "panic", "terrified", "dread", "uneasy", "stressed",
        "stress", "overwhelming", "overwhelmed", "dreading", "phobia",
        "insecure", "unsafe", "threat", "threatened", "pressure",
        "cant handle", "cannot handle", "too much", "losing control"
    ],
    "joy": [
        "happy", "joy", "excited", "great", "amazing", "good", "cheerful",
        "wonderful", "fantastic", "delighted", "thrilled", "blessed",
        "grateful", "content", "pleased", "proud", "elated", "ecstatic",
        "awesome", "excellent", "brilliant", "glad", "motivated", "motivation",
        "ready", "confident", "relaxed", "peaceful", "calm", "refreshed",
        "fresh", "positive", "optimistic", "energetic", "enthusiastic",
        "inspired", "inspiring", "accomplished", "achievement", "succeeded",
        "success", "productive", "fun", "enjoyed", "enjoying", "laughing",
        "smiled", "smiling", "better", "best day", "good day", "great day",
        "wonderful day", "beautiful day", "love my life", "feeling good",
        "feeling great", "feeling amazing", "feeling wonderful", "feel good",
        "feel great", "feel amazing", "hope", "hopeful", "looking forward"
    ],
    "love": [
        "love", "care", "affection", "adore", "cherish", "devoted",
        "romantic", "compassion", "warmth", "fond", "attached", "caring",
        "supportive", "together", "bonding", "friendship", "friends",
        "family", "close to", "mean a lot"
    ],
    "disgust": [
        "disgusting", "gross", "nasty", "revolting", "repulsive",
        "sick", "sickening", "yuck", "eww", "filthy", "vile", "creepy", "disgust"
    ],
    "neutral": [
        "okay", "ok", "fine", "alright", "normal", "usual", "whatever",
        "so so", "meh", "average", "moderate", "nothing special",
        "just another", "regular", "ordinary", "don't know", "dont know", "confused", "not sure", "uncertain", "what is happening"
    ]
}

negative_override_phrases = [
    "feel like dying", "feel like i am dying", "feel like im dying",
    "want to die", "wish i was dead", "can't go on", "cannot go on",
    "bad day", "really bad day", "terrible day", "horrible day",
    "not feeling well", "not feeling good", "feeling terrible",
    "feeling awful", "feeling horrible", "feeling empty", "feeling lost",
    "feeling broken", "feeling hopeless", "feeling helpless",
    "cant handle", "cannot handle", "too much pressure"
]

def keyword_override(text):
    text_lower = text.lower()

    for phrase in negative_override_phrases:
        if phrase in text_lower:
            return "sadness"

    negative_emotions = ["sadness", "anger", "fear", "disgust"]
    positive_emotions = ["joy", "love"]

    emotion_counts = {}
    for emotion, keywords in keyword_map.items():
        count = sum(1 for word in keywords if word in text_lower)
        if count > 0:
            emotion_counts[emotion] = count

    if not emotion_counts:
        return None

    negative_total = sum(c for e, c in emotion_counts.items() if e in negative_emotions)
    positive_total = sum(c for e, c in emotion_counts.items() if e in positive_emotions)

    if negative_total > positive_total:
        negative_matches = {e: c for e, c in emotion_counts.items() if e in negative_emotions}
        return max(negative_matches, key=negative_matches.get)

    if positive_total > negative_total:
        positive_matches = {e: c for e, c in emotion_counts.items() if e in positive_emotions}
        return max(positive_matches, key=positive_matches.get)

    return max(emotion_counts, key=emotion_counts.get)


def keyword_vote_emotion(text):
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in keyword_map.items():
        count = sum(1 for word in keywords if word in text_lower)
        if count > 0:
            scores[emotion] = count
    if not scores:
        return None
    return max(scores, key=scores.get)
