import pandas as pd
import numpy as np

from rouge_score import rouge_scorer


# =========================
# Load Results
# =========================

textrank = pd.read_csv(
    r"D:\NLP_Member 1 news summerize\data\textrank_results.csv"
)


cnn = pd.read_csv(
    r"D:\NLP_Member 1 news summerize\data\cnn_results.csv"
)



scorer = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL"
    ],
    use_stemmer=True
)



def calculate_scores(data):

    rouge1 = []
    rouge2 = []
    rougeL = []


    for actual, generated in zip(
        data["actual_summary"],
        data["generated_summary"]
    ):

        score = scorer.score(
            actual,
            generated
        )

        rouge1.append(
            score["rouge1"].fmeasure
        )

        rouge2.append(
            score["rouge2"].fmeasure
        )

        rougeL.append(
            score["rougeL"].fmeasure
        )


    return [
        np.mean(rouge1),
        np.mean(rouge2),
        np.mean(rougeL)
    ]



# Calculate

textrank_scores = calculate_scores(textrank)

cnn_scores = calculate_scores(cnn)



# Create comparison table

comparison = pd.DataFrame({

    "Model":
    [
        "TextRank",
        "CNN"
    ],

    "ROUGE-1":
    [
        textrank_scores[0],
        cnn_scores[0]
    ],

    "ROUGE-2":
    [
        textrank_scores[1],
        cnn_scores[1]
    ],

    "ROUGE-L":
    [
        textrank_scores[2],
        cnn_scores[2]
    ]

})



print(comparison)



comparison.to_csv(
    r"D:\NLP_Member 1 news summerize\data\model_comparison.csv",
    index=False
)


print("\nComparison saved successfully!")