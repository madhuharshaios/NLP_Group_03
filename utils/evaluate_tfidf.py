import os
import sys
import pandas as pd
from rouge_score import rouge_scorer


# =========================================================
# PROJECT ROOT PATH
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
UTILS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(UTILS_FOLDER)

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# =========================================================
# IMPORT TF-IDF SUMMARIZER
# =========================================================

from utils.tfidf_sentence_ranking import TFIDFSummarizer


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "clean_news_summary.csv"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "metrics"
)

RESULT_PATH = os.path.join(
    OUTPUT_FOLDER,
    "tfidf_rouge_results.csv"
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "summaries",
    "tfidf_predictions.csv"
)


# =========================================================
# CHECK DATASET
# =========================================================

print("=" * 70)
print("TF-IDF SENTENCE RANKING EVALUATION")
print("=" * 70)

if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset file not found:")
    print(DATA_PATH)

    raise SystemExit


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

try:

    df = pd.read_csv(
        DATA_PATH
    )

except UnicodeDecodeError:

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1"
    )


print(
    "Dataset Shape:",
    df.shape
)

print(
    "Columns:",
    df.columns.tolist()
)


# =========================================================
# DETECT COLUMNS
# =========================================================

if (
    "text" in df.columns
    and
    "headlines" in df.columns
):

    article_column = "text"
    summary_column = "headlines"


elif (
    "clean_text" in df.columns
    and
    "clean_summary" in df.columns
):

    article_column = "clean_text"
    summary_column = "clean_summary"


else:

    print(
        "\nERROR: Required columns were not found."
    )

    print(
        "Expected:"
    )

    print(
        "text + headlines"
    )

    print(
        "or"
    )

    print(
        "clean_text + clean_summary"
    )

    raise SystemExit


# =========================================================
# CLEAN DATAFRAME
# =========================================================

df = df[
    [
        article_column,
        summary_column
    ]
].copy()

df.columns = [
    "text",
    "headlines"
]

df = df.dropna(
    subset=[
        "text",
        "headlines"
    ]
)

df = df.drop_duplicates(
    subset=[
        "text",
        "headlines"
    ]
)

df["text"] = df["text"].astype(str)
df["headlines"] = df["headlines"].astype(str)

df = df[
    df["text"].str.strip() != ""
]

df = df[
    df["headlines"].str.strip() != ""
]

print(
    "\nDataset after cleaning:",
    df.shape
)


# =========================================================
# OPTIONAL SAMPLE LIMIT
# =========================================================

# For quick testing:
MAX_SAMPLES = 100

# For full dataset evaluation:
# MAX_SAMPLES = None


if MAX_SAMPLES is not None:

    df = df.head(
        MAX_SAMPLES
    )


print(
    "Evaluation Samples:",
    len(df)
)


# =========================================================
# CREATE TF-IDF SUMMARIZER
# =========================================================

summarizer = TFIDFSummarizer()


# =========================================================
# GENERATE SUMMARIES
# =========================================================

print(
    "\nGenerating TF-IDF summaries..."
)

generated_summaries = []


for index, article in enumerate(
    df["text"],
    start=1
):

    try:

        summary = summarizer.summarize(
            article,
            num_sentences=3
        )

    except Exception as error:

        print(
            f"Warning at sample {index}: {error}"
        )

        summary = ""


    generated_summaries.append(
        summary
    )


    print(
        f"Completed {index}/{len(df)}"
    )


df["tfidf_summary"] = (
    generated_summaries
)


# =========================================================
# CREATE ROUGE SCORER
# =========================================================

scorer = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL"
    ],
    use_stemmer=True
)


# =========================================================
# SCORE STORAGE
# =========================================================

rouge1_precision_scores = []
rouge1_recall_scores = []
rouge1_f1_scores = []

rouge2_precision_scores = []
rouge2_recall_scores = []
rouge2_f1_scores = []

rougeL_precision_scores = []
rougeL_recall_scores = []
rougeL_f1_scores = []


# =========================================================
# CALCULATE ROUGE SCORES
# =========================================================

print(
    "\nCalculating ROUGE scores..."
)


for reference, prediction in zip(
    df["headlines"],
    df["tfidf_summary"]
):

    scores = scorer.score(
        str(reference),
        str(prediction)
    )


    rouge1_precision_scores.append(
        scores["rouge1"].precision
    )

    rouge1_recall_scores.append(
        scores["rouge1"].recall
    )

    rouge1_f1_scores.append(
        scores["rouge1"].fmeasure
    )


    rouge2_precision_scores.append(
        scores["rouge2"].precision
    )

    rouge2_recall_scores.append(
        scores["rouge2"].recall
    )

    rouge2_f1_scores.append(
        scores["rouge2"].fmeasure
    )


    rougeL_precision_scores.append(
        scores["rougeL"].precision
    )

    rougeL_recall_scores.append(
        scores["rougeL"].recall
    )

    rougeL_f1_scores.append(
        scores["rougeL"].fmeasure
    )


# =========================================================
# CALCULATE AVERAGES
# =========================================================

rouge1_precision = (
    sum(rouge1_precision_scores)
    / len(rouge1_precision_scores)
)

rouge1_recall = (
    sum(rouge1_recall_scores)
    / len(rouge1_recall_scores)
)

rouge1_f1 = (
    sum(rouge1_f1_scores)
    / len(rouge1_f1_scores)
)


rouge2_precision = (
    sum(rouge2_precision_scores)
    / len(rouge2_precision_scores)
)

rouge2_recall = (
    sum(rouge2_recall_scores)
    / len(rouge2_recall_scores)
)

rouge2_f1 = (
    sum(rouge2_f1_scores)
    / len(rouge2_f1_scores)
)


rougeL_precision = (
    sum(rougeL_precision_scores)
    / len(rougeL_precision_scores)
)

rougeL_recall = (
    sum(rougeL_recall_scores)
    / len(rougeL_recall_scores)
)

rougeL_f1 = (
    sum(rougeL_f1_scores)
    / len(rougeL_f1_scores)
)


average_rouge_f1 = (
    rouge1_f1
    +
    rouge2_f1
    +
    rougeL_f1
) / 3


# =========================================================
# DISPLAY RESULTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TF-IDF SENTENCE RANKING RESULTS"
)

print(
    "=" * 70
)


print(
    "\nROUGE-1"
)

print(
    f"Precision : {rouge1_precision:.4f}"
)

print(
    f"Recall    : {rouge1_recall:.4f}"
)

print(
    f"F1 Score  : {rouge1_f1:.4f}"
)


print(
    "\nROUGE-2"
)

print(
    f"Precision : {rouge2_precision:.4f}"
)

print(
    f"Recall    : {rouge2_recall:.4f}"
)

print(
    f"F1 Score  : {rouge2_f1:.4f}"
)


print(
    "\nROUGE-L"
)

print(
    f"Precision : {rougeL_precision:.4f}"
)

print(
    f"Recall    : {rougeL_recall:.4f}"
)

print(
    f"F1 Score  : {rougeL_f1:.4f}"
)


print(
    "\nAverage ROUGE F1:"
)

print(
    f"{average_rouge_f1:.4f}"
)


# =========================================================
# SAVE METRIC RESULTS
# =========================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


results_df = pd.DataFrame(
    {
        "Metric": [
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "Average ROUGE F1"
        ],

        "Precision": [
            rouge1_precision,
            rouge2_precision,
            rougeL_precision,
            ""
        ],

        "Recall": [
            rouge1_recall,
            rouge2_recall,
            rougeL_recall,
            ""
        ],

        "F1 Score": [
            rouge1_f1,
            rouge2_f1,
            rougeL_f1,
            average_rouge_f1
        ]
    }
)


results_df.to_csv(
    RESULT_PATH,
    index=False
)


# =========================================================
# SAVE GENERATED SUMMARIES
# =========================================================

os.makedirs(
    os.path.dirname(
        PREDICTION_PATH
    ),
    exist_ok=True
)


prediction_df = df[
    [
        "text",
        "headlines",
        "tfidf_summary"
    ]
].copy()


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "EVALUATION COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nMetrics saved to:"
)

print(
    RESULT_PATH
)


print(
    "\nGenerated summaries saved to:"
)

print(
    PREDICTION_PATH
)