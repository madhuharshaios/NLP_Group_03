import streamlit as st
import nltk
import networkx as nx

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download tokenizer
nltk.download("punkt")


# TextRank Function

def textrank_summary(article):

    sentences = sent_tokenize(article)

    if len(sentences) <= 3:
        return article


    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(sentences)


    similarity = cosine_similarity(tfidf)


    graph = nx.from_numpy_array(similarity)


    scores = nx.pagerank(graph)


    ranked = sorted(
        (
            (scores[i], sentence)
            for i, sentence in enumerate(sentences)
        ),
        reverse=True
    )


    summary = " ".join(
        [
            sentence
            for score, sentence in ranked[:3]
        ]
    )


    return summary



# Interface

st.title("News Summarization System")

st.write(
    "TextRank and CNN based News Summarizer"
)


article = st.text_area(
    "Enter News Article"
)


model = st.selectbox(
    "Select Model",
    [
        "TextRank"
    ]
)


if st.button("Generate Summary"):

    if article:

        if model == "TextRank":

            summary = textrank_summary(article)


        st.subheader("Generated Summary")

        st.write(summary)

    else:

        st.warning(
            "Please enter an article"
        )