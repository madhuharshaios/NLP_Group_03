import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------------------------
# 1. Project paths
# -------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)
MODELS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(MODELS_FOLDER)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "clean_news_summary.csv"
)

SAVED_MODELS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

os.makedirs(SAVED_MODELS_FOLDER, exist_ok=True)


# -------------------------------------------------
# 2. Configuration
# -------------------------------------------------

MAX_ARTICLE_LENGTH = 300
MAX_SUMMARY_LENGTH = 20
MAX_VOCAB_SIZE = 10000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# -------------------------------------------------
# 3. Load dataset
# -------------------------------------------------

print("=" * 60)
print("Loading Clean Dataset")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print("\nError: clean_news_summary.csv file එක හමු නොවුණා.")
    print("මුලින් preprocessing script එක run කරන්න.")
    raise SystemExit


required_columns = ["text", "headlines"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nMissing columns:", missing_columns)
    raise SystemExit


df = df.dropna(subset=required_columns)
df = df.drop_duplicates(subset=required_columns)

df["text"] = df["text"].astype(str)
df["headlines"] = df["headlines"].astype(str)

print("Dataset Shape:", df.shape)


# -------------------------------------------------
# 4. Add start and end tokens
# -------------------------------------------------

df["decoder_input_text"] = (
    "sostok " + df["headlines"]
)

df["decoder_target_text"] = (
    df["headlines"] + " eostok"
)

print("\nSample Article:")
print(df["text"].iloc[0])

print("\nSample Decoder Input:")
print(df["decoder_input_text"].iloc[0])

print("\nSample Decoder Target:")
print(df["decoder_target_text"].iloc[0])


# -------------------------------------------------
# 5. Train-test split
# -------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTraining Samples:", len(train_df))
print("Testing Samples :", len(test_df))


# -------------------------------------------------
# 6. Encoder tokenizer
# -------------------------------------------------

encoder_tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

encoder_tokenizer.fit_on_texts(
    train_df["text"]
)

encoder_train_sequences = (
    encoder_tokenizer.texts_to_sequences(
        train_df["text"]
    )
)

encoder_test_sequences = (
    encoder_tokenizer.texts_to_sequences(
        test_df["text"]
    )
)

encoder_train_data = pad_sequences(
    encoder_train_sequences,
    maxlen=MAX_ARTICLE_LENGTH,
    padding="post",
    truncating="post"
)

encoder_test_data = pad_sequences(
    encoder_test_sequences,
    maxlen=MAX_ARTICLE_LENGTH,
    padding="post",
    truncating="post"
)


# -------------------------------------------------
# 7. Decoder tokenizer
# -------------------------------------------------

decoder_tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

decoder_tokenizer.fit_on_texts(
    train_df["decoder_input_text"].tolist()
    + train_df["decoder_target_text"].tolist()
)

decoder_input_train_sequences = (
    decoder_tokenizer.texts_to_sequences(
        train_df["decoder_input_text"]
    )
)

decoder_target_train_sequences = (
    decoder_tokenizer.texts_to_sequences(
        train_df["decoder_target_text"]
    )
)

decoder_input_test_sequences = (
    decoder_tokenizer.texts_to_sequences(
        test_df["decoder_input_text"]
    )
)

decoder_target_test_sequences = (
    decoder_tokenizer.texts_to_sequences(
        test_df["decoder_target_text"]
    )
)


decoder_input_train_data = pad_sequences(
    decoder_input_train_sequences,
    maxlen=MAX_SUMMARY_LENGTH,
    padding="post",
    truncating="post"
)

decoder_target_train_data = pad_sequences(
    decoder_target_train_sequences,
    maxlen=MAX_SUMMARY_LENGTH,
    padding="post",
    truncating="post"
)

decoder_input_test_data = pad_sequences(
    decoder_input_test_sequences,
    maxlen=MAX_SUMMARY_LENGTH,
    padding="post",
    truncating="post"
)

decoder_target_test_data = pad_sequences(
    decoder_target_test_sequences,
    maxlen=MAX_SUMMARY_LENGTH,
    padding="post",
    truncating="post"
)


# -------------------------------------------------
# 8. Add final dimension to decoder target
# -------------------------------------------------

decoder_target_train_data = np.expand_dims(
    decoder_target_train_data,
    axis=-1
)

decoder_target_test_data = np.expand_dims(
    decoder_target_test_data,
    axis=-1
)


# -------------------------------------------------
# 9. Print data shapes
# -------------------------------------------------

print("\n" + "=" * 60)
print("Prepared Data Shapes")
print("=" * 60)

print(
    "Encoder Train:",
    encoder_train_data.shape
)

print(
    "Decoder Input Train:",
    decoder_input_train_data.shape
)

print(
    "Decoder Target Train:",
    decoder_target_train_data.shape
)

print(
    "Encoder Test:",
    encoder_test_data.shape
)

print(
    "Decoder Input Test:",
    decoder_input_test_data.shape
)

print(
    "Decoder Target Test:",
    decoder_target_test_data.shape
)


# -------------------------------------------------
# 10. Vocabulary sizes
# -------------------------------------------------

encoder_vocab_size = min(
    MAX_VOCAB_SIZE,
    len(encoder_tokenizer.word_index) + 1
)

decoder_vocab_size = min(
    MAX_VOCAB_SIZE,
    len(decoder_tokenizer.word_index) + 1
)

print("\nEncoder Vocabulary Size:", encoder_vocab_size)
print("Decoder Vocabulary Size:", decoder_vocab_size)


# -------------------------------------------------
# 11. Save tokenizers
# -------------------------------------------------

encoder_tokenizer_path = os.path.join(
    SAVED_MODELS_FOLDER,
    "encoder_tokenizer.pkl"
)

decoder_tokenizer_path = os.path.join(
    SAVED_MODELS_FOLDER,
    "decoder_tokenizer.pkl"
)

with open(
    encoder_tokenizer_path,
    "wb"
) as file:
    pickle.dump(
        encoder_tokenizer,
        file
    )

with open(
    decoder_tokenizer_path,
    "wb"
) as file:
    pickle.dump(
        decoder_tokenizer,
        file
    )


# -------------------------------------------------
# 12. Save prepared arrays
# -------------------------------------------------

prepared_data_path = os.path.join(
    SAVED_MODELS_FOLDER,
    "lstm_prepared_data.npz"
)

np.savez_compressed(
    prepared_data_path,

    encoder_train_data=encoder_train_data,
    encoder_test_data=encoder_test_data,

    decoder_input_train_data=decoder_input_train_data,
    decoder_input_test_data=decoder_input_test_data,

    decoder_target_train_data=decoder_target_train_data,
    decoder_target_test_data=decoder_target_test_data,

    encoder_vocab_size=encoder_vocab_size,
    decoder_vocab_size=decoder_vocab_size,

    max_article_length=MAX_ARTICLE_LENGTH,
    max_summary_length=MAX_SUMMARY_LENGTH
)


# -------------------------------------------------
# 13. Save test references
# -------------------------------------------------

test_reference_path = os.path.join(
    SAVED_MODELS_FOLDER,
    "lstm_test_references.csv"
)

test_df[
    ["text", "headlines"]
].to_csv(
    test_reference_path,
    index=False
)


# -------------------------------------------------
# 14. Final output
# -------------------------------------------------

print("\n" + "=" * 60)
print("LSTM Data Preparation Completed Successfully")
print("=" * 60)

print("\nSaved Files:")

print(encoder_tokenizer_path)
print(decoder_tokenizer_path)
print(prepared_data_path)
print(test_reference_path)