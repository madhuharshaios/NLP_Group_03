import os
import re
import csv
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer


# =========================================================
# 1. Project paths
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
UTILS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(UTILS_FOLDER)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "clean_news_summary.csv"
)

SAVED_MODELS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "lsa"
)

TFIDF_MODEL_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "tfidf_vectorizer.pkl"
)

LSA_MODEL_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "lsa_model.pkl"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lsa_predictions.csv"
)

ROUGE_RESULTS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lsa_rouge_scores.csv"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =========================================================
# 2. Configuration
# =========================================================

# මුලින් samples 100ක් evaluate කරන්න.
# Full dataset එක evaluate කරන්න None දාන්න.
MAX_EVALUATION_SAMPLES = 100

# Summary එකට තෝරන sentence ගණන
NUMBER_OF_SENTENCES = 1


# =========================================================
# 3. Check required files
# =========================================================

required_files = [
    DATA_PATH,
    TFIDF_MODEL_PATH,
    LSA_MODEL_PATH
]

missing_files = [
    file_path
    for file_path in required_files
    if not os.path.exists(file_path)
]

if missing_files:
    print("=" * 70)
    print("ERROR: Required files are missing")
    print("=" * 70)

    for file_path in missing_files:
        print(file_path)

    print("\nRun these commands first:")
    print("python preprocessing/preprocess.py")
    print("python models/train_lsa.py")

    raise SystemExit


# =========================================================
# 4. Load trained LSA models
# =========================================================

print("=" * 70)
print("Loading LSA Model")
print("=" * 70)

try:
    tfidf_vectorizer = joblib.load(
        TFIDF_MODEL_PATH
    )

    lsa_model = joblib.load(
        LSA_MODEL_PATH
    )

except Exception as error:
    print("\nFailed to load LSA model files.")
    print("Error:", error)
    print("\nRun the LSA training script again:")
    print("python models/train_lsa.py")
    raise SystemExit

print("TF-IDF vectorizer loaded successfully.")
print("LSA model loaded successfully.")


# =========================================================
# 5. Load dataset
# =========================================================

print("\n" + "=" * 70)
print("Loading Clean Dataset")
print("=" * 70)

try:
    df = pd.read_csv(
        DATA_PATH
    )

except Exception as error:
    print("Failed to load dataset.")
    print("Error:", error)
    raise SystemExit


required_columns = [
    "text",
    "headlines"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nDataset is missing these columns:")
    print(missing_columns)
    raise SystemExit


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

if MAX_EVALUATION_SAMPLES is not None:
    df = df.head(
        MAX_EVALUATION_SAMPLES
    )

print("Evaluation dataset shape:", df.shape)


# =========================================================
# 6. Basic text cleaning
# =========================================================

def clean_text(text):
    """
    Cleans a sentence while keeping only useful English words.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# 7. Split article into sentences
# =========================================================

def split_into_sentences(article):
    """
    Splits an article using punctuation marks.
    """

    if not isinstance(article, str):
        article = str(article)

    article = article.strip()

    if not article:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        article
    )

    valid_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) >= 3:
            valid_sentences.append(
                sentence
            )

    if not valid_sentences and article:
        valid_sentences.append(
            article
        )

    return valid_sentences


# =========================================================
# 8. Generate LSA summary
# =========================================================

def generate_lsa_summary(
    article,
    number_of_sentences=1
):
    """
    Generates an extractive summary using TF-IDF and LSA.
    """

    original_sentences = split_into_sentences(
        article
    )

    if not original_sentences:
        return ""

    if len(original_sentences) <= number_of_sentences:
        return " ".join(
            original_sentences
        )

    cleaned_sentences = [
        clean_text(sentence)
        for sentence in original_sentences
    ]

    valid_indices = [
        index
        for index, sentence in enumerate(cleaned_sentences)
        if sentence.strip()
    ]

    if not valid_indices:
        return original_sentences[0]

    valid_cleaned_sentences = [
        cleaned_sentences[index]
        for index in valid_indices
    ]

    try:
        sentence_tfidf_matrix = (
            tfidf_vectorizer.transform(
                valid_cleaned_sentences
            )
        )

        sentence_lsa_matrix = (
            lsa_model.transform(
                sentence_tfidf_matrix
            )
        )

    except Exception:
        return original_sentences[0]

    if sentence_lsa_matrix.shape[0] == 0:
        return original_sentences[0]

    document_vector = np.mean(
        sentence_lsa_matrix,
        axis=0
    ).reshape(
        1,
        -1
    )

    similarity_scores = cosine_similarity(
        sentence_lsa_matrix,
        document_vector
    ).flatten()

    number_to_select = min(
        number_of_sentences,
        len(similarity_scores)
    )

    selected_relative_indices = np.argsort(
        similarity_scores
    )[-number_to_select:]

    selected_original_indices = [
        valid_indices[index]
        for index in selected_relative_indices
    ]

    selected_original_indices.sort()

    selected_sentences = [
        original_sentences[index]
        for index in selected_original_indices
    ]

    return " ".join(
        selected_sentences
    )


# =========================================================
# 9. Create ROUGE scorer
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
# 10. Evaluation score storage
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

prediction_rows = []


# =========================================================
# 11. Generate predictions and calculate ROUGE
# =========================================================

print("\n" + "=" * 70)
print("Starting LSA ROUGE Evaluation")
print("=" * 70)

total_samples = len(df)

for completed, (_, row) in enumerate(
    df.iterrows(),
    start=1
):
    article = str(
        row["text"]
    )

    reference_summary = str(
        row["headlines"]
    )

    generated_summary = generate_lsa_summary(
        article,
        number_of_sentences=NUMBER_OF_SENTENCES
    )

    if not generated_summary.strip():
        generated_summary = ""

    scores = scorer.score(
        reference_summary,
        generated_summary
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

    prediction_rows.append({
        "article": article,
        "reference_summary": reference_summary,
        "generated_summary": generated_summary,
        "rouge1_precision": scores["rouge1"].precision,
        "rouge1_recall": scores["rouge1"].recall,
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_f1": scores["rougeL"].fmeasure
    })

    print(
        f"Completed {completed}/{total_samples}"
    )


# =========================================================
# 12. Check evaluation results
# =========================================================

if not prediction_rows:
    print("\nNo predictions were generated.")
    raise SystemExit


# =========================================================
# 13. Calculate average ROUGE scores
# =========================================================

average_rouge1_precision = float(
    np.mean(rouge1_precision_scores)
)

average_rouge1_recall = float(
    np.mean(rouge1_recall_scores)
)

average_rouge1_f1 = float(
    np.mean(rouge1_f1_scores)
)


average_rouge2_precision = float(
    np.mean(rouge2_precision_scores)
)

average_rouge2_recall = float(
    np.mean(rouge2_recall_scores)
)

average_rouge2_f1 = float(
    np.mean(rouge2_f1_scores)
)


average_rougeL_precision = float(
    np.mean(rougeL_precision_scores)
)

average_rougeL_recall = float(
    np.mean(rougeL_recall_scores)
)

average_rougeL_f1 = float(
    np.mean(rougeL_f1_scores)
)


# =========================================================
# 14. Display average results
# =========================================================

print("\n" + "=" * 70)
print("Average LSA ROUGE Scores")
print("=" * 70)

print("\nROUGE-1")
print(
    f"Precision: {average_rouge1_precision:.4f}"
)
print(
    f"Recall   : {average_rouge1_recall:.4f}"
)
print(
    f"F1 Score : {average_rouge1_f1:.4f}"
)

print("\nROUGE-2")
print(
    f"Precision: {average_rouge2_precision:.4f}"
)
print(
    f"Recall   : {average_rouge2_recall:.4f}"
)
print(
    f"F1 Score : {average_rouge2_f1:.4f}"
)

print("\nROUGE-L")
print(
    f"Precision: {average_rougeL_precision:.4f}"
)
print(
    f"Recall   : {average_rougeL_recall:.4f}"
)
print(
    f"F1 Score : {average_rougeL_f1:.4f}"
)


# =========================================================
# 15. Save individual predictions
# =========================================================

predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8"
)

print("\nPredictions saved to:")
print(PREDICTIONS_PATH)


# =========================================================
# 16. Save average ROUGE scores
# =========================================================

rouge_result_rows = [
    [
        "ROUGE-1",
        average_rouge1_precision,
        average_rouge1_recall,
        average_rouge1_f1
    ],
    [
        "ROUGE-2",
        average_rouge2_precision,
        average_rouge2_recall,
        average_rouge2_f1
    ],
    [
        "ROUGE-L",
        average_rougeL_precision,
        average_rougeL_recall,
        average_rougeL_f1
    ]
]

with open(
    ROUGE_RESULTS_PATH,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(
        file
    )

    writer.writerow([
        "Metric",
        "Precision",
        "Recall",
        "F1 Score"
    ])

    writer.writerows(
        rouge_result_rows
    )

print("\nROUGE scores saved to:")
print(ROUGE_RESULTS_PATH)


# =========================================================
# 17. Final output
# =========================================================

print("\n" + "=" * 70)
print("LSA ROUGE Evaluation Completed Successfully")
print("=" * 70)

print("\nGenerated files:")
print(PREDICTIONS_PATH)
print(ROUGE_RESULTS_PATH)