import pandas as pd
import nltk
import networkx as nx

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(r"D:\NLP_Member 1 news summerize\data\clean_news.csv")

article = df["article"][0]

print("Original Article:\n")
print(article)

# Split article into sentences
sentences = sent_tokenize(article)

print("\nNumber of Sentences:", len(sentences))

# TF-IDF
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(sentences)

# Similarity Matrix
similarity_matrix = cosine_similarity(tfidf)

# Graph
graph = nx.from_numpy_array(similarity_matrix)

# PageRank
scores = nx.pagerank(graph)

# Rank sentences
ranked = sorted(
    ((scores[i], s) for i, s in enumerate(sentences)),
    reverse=True
)

summary = " ".join([s for score, s in ranked[:3]])

print("\nGenerated Summary:\n")
print(summary)