import pandas as pd
from rouge_score import rouge_scorer


def evaluate_rouge(reference, prediction):

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, prediction)

    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }


if __name__ == "__main__":

    reference = "Artificial intelligence improves agriculture by increasing crop productivity."

    prediction = "Artificial intelligence increases crop productivity."

    results = evaluate_rouge(reference, prediction)

    df = pd.DataFrame([results])

    print(df)

    df.to_csv(
        "outputs/rouge_scores.csv",
        index=False
    )

    print("\nROUGE scores saved successfully.")