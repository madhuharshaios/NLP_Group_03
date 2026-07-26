import joblib

# Load models
vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")
lsa = joblib.load("saved_models/lsa_model.pkl")


def generate_summary(article, num_sentences=3):

    # Split article into sentences
    sentences = article.split(".")

    if len(sentences) <= num_sentences:
        return article

    # TF-IDF representation
    sentence_vectors = vectorizer.transform(sentences)

    # LSA transformation
    lsa_vectors = lsa.transform(sentence_vectors)

    # Sentence importance score
    scores = lsa_vectors.sum(axis=1)

    # Get top sentence indexes
    ranked = scores.argsort()[::-1]

    selected = sorted(ranked[:num_sentences])

    summary = ". ".join(
        sentences[i].strip()
        for i in selected
        if sentences[i].strip()
    )

    return summary


if __name__ == "__main__":

    article = input("Paste News Article:\n\n")

    print("\nGenerated Summary\n")

    print(generate_summary(article))