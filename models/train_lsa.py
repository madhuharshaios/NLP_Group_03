import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load cleaned dataset
df = pd.read_csv("data/clean_news_summary.csv")

# Input and target
articles = df["text"]
summaries = df["headlines"]

print("Dataset Loaded Successfully")
print(df.shape)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(articles)

print("TF-IDF Shape:", X.shape)

# LSA using Truncated SVD
lsa = TruncatedSVD(
    n_components=100,
    random_state=42
)

X_lsa = lsa.fit_transform(X)

print("LSA Shape:", X_lsa.shape)

# Save models
os.makedirs("saved_models", exist_ok=True)

joblib.dump(vectorizer, "saved_models/tfidf_vectorizer.pkl")
joblib.dump(lsa, "saved_models/lsa_model.pkl")

print("\nModels Saved Successfully")