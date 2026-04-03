import joblib
from preprocess import clean_text

svm_model = joblib.load("saved_models/svm_model.pkl")
nb_model = joblib.load("saved_models/nb_model.pkl")
vectorizer = joblib.load("saved_models/vectorizer.pkl")


def emotion_score(emotion):
    scores = {
        "joy": 1,
        "love": 0.9,
        "surprise": 0.3,
        "neutral": 0,
        "sadness": -1,
        "fear": -0.8,
        "anger": -0.7,
        "disgust": -0.9
    }
    return scores.get(emotion, 0)


def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    svm_pred = str(svm_model.predict(vec)[0])
    nb_pred = str(nb_model.predict(vec)[0])

    return {
        "SVM Prediction": svm_pred,
        "NB Prediction": nb_pred,
        "SVM Score": emotion_score(svm_pred),
        "NB Score": emotion_score(nb_pred)
    }


if __name__ == "__main__":
    text = input("Enter text: ")
    print("\nResult:\n", predict(text))