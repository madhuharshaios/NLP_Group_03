import pandas as pd
import nltk
import networkx as nx

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rouge_score import rouge_scorer


# Load cleaned dataset
df = pd.read_csv(
    r"D:\NLP_Member 1 news summerize\data\clean_news.csv"
)


def textrank_summary(article, sentence_count=3):

    sentences = sent_tokenize(article)

    if len(sentences) <= sentence_count:
        return article

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(graph)

    ranked_sentences = sorted(
        ((scores[i], sentence)
         for i, sentence in enumerate(sentences)),
        reverse=True
    )

    summary = " ".join(
        [sentence for score, sentence in ranked_sentences[:sentence_count]]
    )

    return summary



references = []
predictions = []


# Test first 100 news articles
for i in range(100):

    article = df["article"][i]
    real_summary = df["summary"][i]

    generated_summary = textrank_summary(article)

    references.append(real_summary)
    predictions.append(generated_summary)



# ROUGE evaluation

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)


rouge1 = []
rouge2 = []
rougeL = []


for ref, pred in zip(references, predictions):

    score = scorer.score(ref, pred)

    rouge1.append(score["rouge1"].fmeasure)
    rouge2.append(score["rouge2"].fmeasure)
    rougeL.append(score["rougeL"].fmeasure)



print("Average ROUGE-1:", sum(rouge1)/len(rouge1))
print("Average ROUGE-2:", sum(rouge2)/len(rouge2))
print("Average ROUGE-L:", sum(rougeL)/len(rougeL))


# Save results

results = pd.DataFrame({
    "actual_summary": references,
    "generated_summary": predictions
})


results.to_csv(
    r"D:\NLP_Member 1 news summerize\data\textrank_results.csv",
    index=False
)


print("CSV file saved successfully!")