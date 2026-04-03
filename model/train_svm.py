import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load data
df = pd.read_csv("../data/processed/final_dataset.csv")

X = df['text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔥 Improved SVM
model = LinearSVC(class_weight='balanced', C=2)
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("🔹 SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/svm_model.pkl")
joblib.dump(vectorizer, "saved_models/vectorizer.pkl")

print("\n✅ SVM model saved!")