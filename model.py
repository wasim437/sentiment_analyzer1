import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Sample data (Positive/Negative sentences)
data = {
    "text": ["I love this!", "This is bad.", "Awesome job!", "Worst experience ever."],
    "sentiment": ["positive", "negative", "positive", "negative"]
}
df = pd.DataFrame(data)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
model = LinearSVC()
model.fit(X, df["sentiment"])

# Save model & vectorizer
joblib.dump(model, "sentiment_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")