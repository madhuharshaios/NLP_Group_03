import os
import sys
import csv
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from rouge_score import rouge_scorer


# =========================================================
# 1. Project paths
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
UTILS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(UTILS_FOLDER)

SAVED_MODELS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "lstm"
)

MODEL_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "best_lstm.keras"
)

ENCODER_TOKENIZER_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "encoder_tokenizer.pkl"
)

DECODER_TOKENIZER_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "decoder_tokenizer.pkl"
)

PREPARED_DATA_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "lstm_prepared_data.npz"
)

TEST_REFERENCES_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "lstm_test_references.csv"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lstm_predictions.csv"
)

ROUGE_RESULTS_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lstm_rouge_scores.csv"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =========================================================
# 2. Configuration
# =========================================================

# Full test set එක evaluate කරන්න නම් None දාන්න.
# ඉක්මනින් test කරන්න නම් 100 වගේ value එකක් දාන්න.

MAX_EVALUATION_SAMPLES = 100


# =========================================================
# 3. Check required files
# =========================================================

required_files = [
    MODEL_PATH,
    ENCODER_TOKENIZER_PATH,
    DECODER_TOKENIZER_PATH,
    PREPARED_DATA_PATH,
    TEST_REFERENCES_PATH
]

missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    print("=" * 70)
    print("ERROR: Required files are missing")
    print("=" * 70)

    for file_path in missing_files:
        print(file_path)

    print("\nRun these commands first:")
    print("python models/prepare_lstm_data.py")
    print("python models/train_lstm.py")

    raise SystemExit


# =========================================================
# 4. Load prepared configuration
# =========================================================

prepared_data = np.load(
    PREPARED_DATA_PATH
)

max_article_length = int(
    prepared_data["max_article_length"]
)

max_summary_length = int(
    prepared_data["max_summary_length"]
)


# =========================================================
# 5. Load tokenizers
# =========================================================

with open(
    ENCODER_TOKENIZER_PATH,
    "rb"
) as file:

    encoder_tokenizer = pickle.load(file)


with open(
    DECODER_TOKENIZER_PATH,
    "rb"
) as file:

    decoder_tokenizer = pickle.load(file)


reverse_decoder_word_index = {
    index: word
    for word, index
    in decoder_tokenizer.word_index.items()
}


# =========================================================
# 6. Load trained model
# =========================================================

print("=" * 70)
print("Loading trained LSTM model")
print("=" * 70)

training_model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================================================
# 7. Get trained layers
# =========================================================

encoder_input_layer = training_model.get_layer(
    "encoder_inputs"
)

encoder_embedding_layer = training_model.get_layer(
    "encoder_embedding"
)

encoder_lstm_layer = training_model.get_layer(
    "encoder_lstm"
)

decoder_embedding_layer = training_model.get_layer(
    "decoder_embedding"
)

decoder_lstm_layer = training_model.get_layer(
    "decoder_lstm"
)

decoder_dense_layer = training_model.get_layer(
    "decoder_output"
)


# =========================================================
# 8. Build encoder inference model
# =========================================================

encoder_inputs = encoder_input_layer.output

encoder_embedding_output = encoder_embedding_layer(
    encoder_inputs
)

_, encoder_state_h, encoder_state_c = encoder_lstm_layer(
    encoder_embedding_output
)

encoder_model = Model(
    inputs=encoder_inputs,
    outputs=[
        encoder_state_h,
        encoder_state_c
    ],
    name="encoder_inference_model"
)


# =========================================================
# 9. Build decoder inference model
# =========================================================

latent_dimension = decoder_lstm_layer.units

decoder_single_input = Input(
    shape=(1,),
    name="decoder_single_input"
)

decoder_state_input_h = Input(
    shape=(latent_dimension,),
    name="decoder_state_input_h"
)

decoder_state_input_c = Input(
    shape=(latent_dimension,),
    name="decoder_state_input_c"
)

decoder_embedding_output = decoder_embedding_layer(
    decoder_single_input
)

decoder_output, decoder_state_h, decoder_state_c = (
    decoder_lstm_layer(
        decoder_embedding_output,
        initial_state=[
            decoder_state_input_h,
            decoder_state_input_c
        ]
    )
)

decoder_output = decoder_dense_layer(
    decoder_output
)

decoder_model = Model(
    inputs=[
        decoder_single_input,
        decoder_state_input_h,
        decoder_state_input_c
    ],
    outputs=[
        decoder_output,
        decoder_state_h,
        decoder_state_c
    ],
    name="decoder_inference_model"
)


# =========================================================
# 10. Clean input text
# =========================================================

def clean_input_text(text):
    """
    Performs simple input cleaning.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = " ".join(text.split())

    return text


# =========================================================
# 11. Prepare article
# =========================================================

def prepare_article(article):
    """
    Converts article text into padded sequence.
    """

    cleaned_article = clean_input_text(
        article
    )

    sequence = encoder_tokenizer.texts_to_sequences(
        [cleaned_article]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_article_length,
        padding="post",
        truncating="post"
    )

    return padded_sequence


# =========================================================
# 12. Generate summary
# =========================================================

def generate_summary(article):
    """
    Generates a summary using greedy decoding.
    """

    article_sequence = prepare_article(
        article
    )

    state_h, state_c = encoder_model.predict(
        article_sequence,
        verbose=0
    )

    start_token_index = (
        decoder_tokenizer.word_index.get(
            "sostok"
        )
    )

    end_token_index = (
        decoder_tokenizer.word_index.get(
            "eostok"
        )
    )

    if start_token_index is None:
        raise ValueError(
            "sostok token not found."
        )

    if end_token_index is None:
        raise ValueError(
            "eostok token not found."
        )

    target_sequence = np.zeros(
        (1, 1),
        dtype=np.int32
    )

    target_sequence[0, 0] = start_token_index

    generated_words = []

    for _ in range(max_summary_length):

        output_tokens, state_h, state_c = (
            decoder_model.predict(
                [
                    target_sequence,
                    state_h,
                    state_c
                ],
                verbose=0
            )
        )

        predicted_token_index = int(
            np.argmax(
                output_tokens[0, -1, :]
            )
        )

        if predicted_token_index == 0:
            break

        if predicted_token_index == end_token_index:
            break

        predicted_word = reverse_decoder_word_index.get(
            predicted_token_index
        )

        if predicted_word is None:
            break

        if predicted_word not in [
            "sostok",
            "eostok",
            "<oov>"
        ]:

            generated_words.append(
                predicted_word
            )

        target_sequence = np.zeros(
            (1, 1),
            dtype=np.int32
        )

        target_sequence[0, 0] = (
            predicted_token_index
        )

    return " ".join(generated_words)


# =========================================================
# 13. Load test references
# =========================================================

print("\n" + "=" * 70)
print("Loading test articles and reference summaries")
print("=" * 70)

test_df = pd.read_csv(
    TEST_REFERENCES_PATH
)

test_df = test_df.dropna(
    subset=[
        "text",
        "headlines"
    ]
)

if MAX_EVALUATION_SAMPLES is not None:

    test_df = test_df.head(
        MAX_EVALUATION_SAMPLES
    )

print(
    "Number of evaluation samples:",
    len(test_df)
)


# =========================================================
# 14. Create ROUGE scorer
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
# 15. Evaluation storage
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
# 16. Generate predictions and calculate ROUGE
# =========================================================

print("\n" + "=" * 70)
print("Starting LSTM ROUGE Evaluation")
print("=" * 70)

for index, row in test_df.iterrows():

    article = str(
        row["text"]
    )

    reference_summary = str(
        row["headlines"]
    )

    generated_summary = generate_summary(
        article
    )

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
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure
    })

    completed = len(prediction_rows)

    print(
        f"Completed {completed}/{len(test_df)}"
    )


# =========================================================
# 17. Calculate average scores
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
# 18. Display results
# =========================================================

print("\n" + "=" * 70)
print("Average LSTM ROUGE Scores")
print("=" * 70)

print("\nROUGE-1")
print("Precision:", average_rouge1_precision)
print("Recall   :", average_rouge1_recall)
print("F1 Score :", average_rouge1_f1)

print("\nROUGE-2")
print("Precision:", average_rouge2_precision)
print("Recall   :", average_rouge2_recall)
print("F1 Score :", average_rouge2_f1)

print("\nROUGE-L")
print("Precision:", average_rougeL_precision)
print("Recall   :", average_rougeL_recall)
print("F1 Score :", average_rougeL_f1)


# =========================================================
# 19. Save generated predictions
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
# 20. Save average ROUGE results
# =========================================================

rouge_results = [
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

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Precision",
        "Recall",
        "F1 Score"
    ])

    writer.writerows(
        rouge_results
    )

print("\nROUGE scores saved to:")
print(ROUGE_RESULTS_PATH)


# =========================================================
# 21. Final output
# =========================================================

print("\n" + "=" * 70)
print("LSTM ROUGE Evaluation Completed Successfully")
print("=" * 70)