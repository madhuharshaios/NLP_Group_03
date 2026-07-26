import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 1. Project paths
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
PREDICT_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(PREDICT_FOLDER)

SAVED_MODELS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "saved_models"
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


# =========================================================
# 2. Check required files
# =========================================================

required_files = [
    MODEL_PATH,
    ENCODER_TOKENIZER_PATH,
    DECODER_TOKENIZER_PATH,
    PREPARED_DATA_PATH
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
    print("python models/prepare_lstm_data.py")
    print("python models/train_lstm.py")

    raise SystemExit


# =========================================================
# 3. Load configuration
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

decoder_vocab_size = int(
    prepared_data["decoder_vocab_size"]
)


# =========================================================
# 4. Load tokenizers
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


# Create reverse decoder dictionary

reverse_decoder_word_index = {
    index: word
    for word, index
    in decoder_tokenizer.word_index.items()
}


# =========================================================
# 5. Load trained model
# =========================================================

print("=" * 70)
print("Loading trained LSTM model")
print("=" * 70)

training_model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================================================
# 6. Get trained layers
# =========================================================

encoder_inputs = training_model.get_layer(
    "encoder_inputs"
).output

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
# 7. Build encoder inference model
# =========================================================

encoder_embedding_output = encoder_embedding_layer(
    encoder_inputs
)

encoder_outputs, encoder_state_h, encoder_state_c = (
    encoder_lstm_layer(
        encoder_embedding_output
    )
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
# 8. Build decoder inference model
# =========================================================

decoder_single_input = Input(
    shape=(1,),
    name="decoder_single_input"
)

decoder_state_input_h = Input(
    shape=(
        decoder_lstm_layer.units,
    ),
    name="decoder_state_input_h"
)

decoder_state_input_c = Input(
    shape=(
        decoder_lstm_layer.units,
    ),
    name="decoder_state_input_c"
)

decoder_state_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

decoder_single_embedding = decoder_embedding_layer(
    decoder_single_input
)

decoder_single_output, decoder_output_h, decoder_output_c = (
    decoder_lstm_layer(
        decoder_single_embedding,
        initial_state=decoder_state_inputs
    )
)

decoder_single_output = decoder_dense_layer(
    decoder_single_output
)

decoder_model = Model(
    inputs=[
        decoder_single_input,
        decoder_state_input_h,
        decoder_state_input_c
    ],
    outputs=[
        decoder_single_output,
        decoder_output_h,
        decoder_output_c
    ],
    name="decoder_inference_model"
)


# =========================================================
# 9. Clean input text
# =========================================================

def clean_input_text(text):
    """
    Performs basic cleaning for prediction input.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = " ".join(text.split())

    return text


# =========================================================
# 10. Convert article to sequence
# =========================================================

def prepare_article(article):
    """
    Converts an article into the padded encoder sequence.
    """

    cleaned_article = clean_input_text(
        article
    )

    article_sequence = (
        encoder_tokenizer.texts_to_sequences(
            [cleaned_article]
        )
    )

    article_sequence = pad_sequences(
        article_sequence,
        maxlen=max_article_length,
        padding="post",
        truncating="post"
    )

    return article_sequence


# =========================================================
# 11. Generate summary
# =========================================================

def generate_summary(article):
    """
    Generates a summary word by word using greedy decoding.
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
            "'sostok' token was not found "
            "in the decoder tokenizer."
        )

    if end_token_index is None:
        raise ValueError(
            "'eostok' token was not found "
            "in the decoder tokenizer."
        )

    target_sequence = np.zeros(
        (1, 1),
        dtype=np.int32
    )

    target_sequence[0, 0] = (
        start_token_index
    )

    generated_words = []

    for _ in range(
        max_summary_length
    ):

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
                output_tokens[
                    0,
                    -1,
                    :
                ]
            )
        )

        if predicted_token_index == 0:
            break

        if predicted_token_index == end_token_index:
            break

        predicted_word = (
            reverse_decoder_word_index.get(
                predicted_token_index
            )
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

    generated_summary = " ".join(
        generated_words
    )

    if not generated_summary.strip():
        return (
            "The model could not generate "
            "a meaningful summary."
        )

    return generated_summary


# =========================================================
# 12. Command line prediction
# =========================================================

def main():

    print("\n" + "=" * 70)
    print("LSTM Text Summarization")
    print("=" * 70)

    print(
        "\nEnter or paste a news article below."
    )

    print(
        "After entering the article, press Enter."
    )

    article = input(
        "\nArticle: "
    ).strip()

    if not article:
        print(
            "\nError: Article cannot be empty."
        )
        return

    try:

        summary = generate_summary(
            article
        )

        print("\n" + "=" * 70)
        print("Generated LSTM Summary")
        print("=" * 70)

        print(summary)

    except Exception as error:

        print("\nPrediction failed:")
        print(str(error))


# =========================================================
# 13. Run program
# =========================================================

if __name__ == "__main__":
    main()