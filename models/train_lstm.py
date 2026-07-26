import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)


# =================================================
# 1. Project paths
# =================================================

CURRENT_FILE = os.path.abspath(__file__)
MODELS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(MODELS_FOLDER)

SAVED_MODELS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "lstm"
)

PREPARED_DATA_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "lstm_prepared_data.npz"
)

MODEL_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "lstm.keras"
)

BEST_MODEL_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "best_lstm.keras"
)

HISTORY_CSV_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lstm_training_history.csv"
)

ACCURACY_GRAPH_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lstm_accuracy.png"
)

LOSS_GRAPH_PATH = os.path.join(
    OUTPUT_FOLDER,
    "lstm_loss.png"
)

os.makedirs(
    SAVED_MODELS_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =================================================
# 2. Configuration
# =================================================

EMBEDDING_DIM = 128
LATENT_DIM = 256

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001


# =================================================
# 3. Check prepared data
# =================================================

if not os.path.exists(PREPARED_DATA_PATH):
    print("=" * 60)
    print("ERROR: Prepared LSTM data file එක හමු නොවුණා.")
    print("=" * 60)

    print("\nමුලින් මේ command එක run කරන්න:")
    print("python models/prepare_lstm_data.py")

    raise SystemExit


# =================================================
# 4. Load prepared data
# =================================================

print("=" * 60)
print("Loading Prepared LSTM Data")
print("=" * 60)

data = np.load(
    PREPARED_DATA_PATH
)

encoder_train_data = data[
    "encoder_train_data"
]

decoder_input_train_data = data[
    "decoder_input_train_data"
]

decoder_target_train_data = data[
    "decoder_target_train_data"
]

encoder_test_data = data[
    "encoder_test_data"
]

decoder_input_test_data = data[
    "decoder_input_test_data"
]

decoder_target_test_data = data[
    "decoder_target_test_data"
]

encoder_vocab_size = int(
    data["encoder_vocab_size"]
)

decoder_vocab_size = int(
    data["decoder_vocab_size"]
)

max_article_length = int(
    data["max_article_length"]
)

max_summary_length = int(
    data["max_summary_length"]
)


print("\nEncoder Train Shape:")
print(encoder_train_data.shape)

print("\nDecoder Input Train Shape:")
print(decoder_input_train_data.shape)

print("\nDecoder Target Train Shape:")
print(decoder_target_train_data.shape)

print("\nEncoder Test Shape:")
print(encoder_test_data.shape)

print("\nDecoder Input Test Shape:")
print(decoder_input_test_data.shape)

print("\nDecoder Target Test Shape:")
print(decoder_target_test_data.shape)

print("\nEncoder Vocabulary Size:")
print(encoder_vocab_size)

print("\nDecoder Vocabulary Size:")
print(decoder_vocab_size)


# =================================================
# 5. Build encoder
# =================================================

encoder_inputs = Input(
    shape=(max_article_length,),
    name="encoder_inputs"
)

encoder_embedding_layer = Embedding(
    input_dim=encoder_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="encoder_embedding"
)

encoder_embedding_output = encoder_embedding_layer(
    encoder_inputs
)

encoder_lstm_layer = LSTM(
    LATENT_DIM,
    return_state=True,
    name="encoder_lstm"
)

encoder_output, encoder_state_h, encoder_state_c = (
    encoder_lstm_layer(
        encoder_embedding_output
    )
)

encoder_states = [
    encoder_state_h,
    encoder_state_c
]


# =================================================
# 6. Build decoder
# =================================================

decoder_inputs = Input(
    shape=(max_summary_length,),
    name="decoder_inputs"
)

decoder_embedding_layer = Embedding(
    input_dim=decoder_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="decoder_embedding"
)

decoder_embedding_output = decoder_embedding_layer(
    decoder_inputs
)

decoder_lstm_layer = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)

decoder_outputs, decoder_state_h, decoder_state_c = (
    decoder_lstm_layer(
        decoder_embedding_output,
        initial_state=encoder_states
    )
)

decoder_dense_layer = Dense(
    decoder_vocab_size,
    activation="softmax",
    name="decoder_output"
)

decoder_outputs = decoder_dense_layer(
    decoder_outputs
)


# =================================================
# 7. Build complete model
# =================================================

model = Model(
    inputs=[
        encoder_inputs,
        decoder_inputs
    ],
    outputs=decoder_outputs,
    name="seq2seq_lstm_summarizer"
)


# =================================================
# 8. Compile model
# =================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n" + "=" * 60)
print("LSTM Model Summary")
print("=" * 60)

model.summary()


# =================================================
# 9. Callbacks
# =================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

csv_logger = CSVLogger(
    HISTORY_CSV_PATH,
    append=False
)


# =================================================
# 10. Train model
# =================================================

print("\n" + "=" * 60)
print("Starting LSTM Model Training")
print("=" * 60)

history = model.fit(
    x=[
        encoder_train_data,
        decoder_input_train_data
    ],

    y=decoder_target_train_data,

    validation_data=(
        [
            encoder_test_data,
            decoder_input_test_data
        ],
        decoder_target_test_data
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_learning_rate,
        csv_logger
    ],

    verbose=1
)


# =================================================
# 11. Evaluate model
# =================================================

print("\n" + "=" * 60)
print("Evaluating LSTM Model")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    [
        encoder_test_data,
        decoder_input_test_data
    ],

    decoder_target_test_data,

    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest Loss:")
print(test_loss)

print("\nTest Accuracy:")
print(test_accuracy)


# =================================================
# 12. Save final model
# =================================================

model.save(
    MODEL_PATH
)

print("\nFinal model saved successfully:")
print(MODEL_PATH)

print("\nBest model saved successfully:")
print(BEST_MODEL_PATH)


# =================================================
# 13. Save evaluation results
# =================================================

evaluation_path = os.path.join(
    OUTPUT_FOLDER,
    "lstm_evaluation.csv"
)

with open(
    evaluation_path,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Test Loss",
        test_loss
    ])

    writer.writerow([
        "Test Accuracy",
        test_accuracy
    ])

print("\nEvaluation results saved:")
print(evaluation_path)


# =================================================
# 14. Accuracy graph
# =================================================

training_accuracy = history.history.get(
    "accuracy",
    []
)

validation_accuracy = history.history.get(
    "val_accuracy",
    []
)

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    training_accuracy,
    label="Training Accuracy"
)

plt.plot(
    validation_accuracy,
    label="Validation Accuracy"
)

plt.title(
    "LSTM Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("\nAccuracy graph saved:")
print(ACCURACY_GRAPH_PATH)


# =================================================
# 15. Loss graph
# =================================================

training_loss = history.history.get(
    "loss",
    []
)

validation_loss = history.history.get(
    "val_loss",
    []
)

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    training_loss,
    label="Training Loss"
)

plt.plot(
    validation_loss,
    label="Validation Loss"
)

plt.title(
    "LSTM Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("\nLoss graph saved:")
print(LOSS_GRAPH_PATH)


# =================================================
# 16. Final output
# =================================================

print("\n" + "=" * 60)
print("LSTM Training Completed Successfully")
print("=" * 60)

print("\nGenerated Files:")

print(MODEL_PATH)
print(BEST_MODEL_PATH)
print(HISTORY_CSV_PATH)
print(evaluation_path)
print(ACCURACY_GRAPH_PATH)
print(LOSS_GRAPH_PATH)