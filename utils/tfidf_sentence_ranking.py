import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")


class TFIDFSummarizer:

    def __init__(self):
        pass


    def summarize(self, article, num_sentences=3):

        # Split article into sentences
        sentences = nltk.sent_tokenize(article)

        # Small articles
        if len(sentences) <= num_sentences:
            return article

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Sentence Similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        scores = []

        for i in range(len(sentences)):

            score = similarity_matrix[i].sum()

            scores.append(score)

        # Sort
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Select Top Sentences
        selected = sorted(
            ranked[:num_sentences],
            key=lambda x: x[0]
        )

        summary = " ".join(
            [sentences[index] for index, score in selected]
        )

        return summary


    def summarize_dataset(self, dataframe):

        summaries = []

        print("Generating TF-IDF Summaries...")

        for article in dataframe["text"]:

            summaries.append(
                self.summarize(article)
            )

        dataframe["tfidf_summary"] = summaries

        return dataframe