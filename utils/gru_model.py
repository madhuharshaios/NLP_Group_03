import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GRU,
    Dense
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# =========================================================
# GRU SUMMARIZER
# =========================================================

class GRUSummarizer:

    def __init__(self):

        # ---------------------------------------------
        # Hyperparameters
        # ---------------------------------------------

        self.max_words = 10000
        self.max_article_length = 300
        self.max_summary_length = 20

        self.embedding_dim = 128
        self.gru_units = 256

        self.batch_size = 16
        self.epochs = 20
        self.learning_rate = 0.001


        # ---------------------------------------------
        # Tokenizers
        # ---------------------------------------------

        self.article_tokenizer = Tokenizer(
            num_words=self.max_words,
            oov_token="<OOV>"
        )

        self.summary_tokenizer = Tokenizer(
            num_words=self.max_words,
            oov_token="<OOV>"
        )


    # =====================================================
    # LOAD DATASET
    # =====================================================

    def load_dataset(self, path):

        print("=" * 70)
        print("Loading Dataset...")
        print("=" * 70)

        if not os.path.exists(path):

            print("\nERROR: Dataset file not found:")
            print(path)

            raise SystemExit


        # ---------------------------------------------
        # Load CSV
        # ---------------------------------------------

        try:

            self.df = pd.read_csv(path)

        except UnicodeDecodeError:

            self.df = pd.read_csv(
                path,
                encoding="latin-1"
            )


        print("\nDataset loaded successfully.")

        print(
            "Original Dataset Shape:",
            self.df.shape
        )

        print(
            "\nAvailable Columns:"
        )

        print(
            self.df.columns.tolist()
        )


        # ---------------------------------------------
        # Detect article and summary columns
        # ---------------------------------------------

        if (
            "text" in self.df.columns
            and
            "headlines" in self.df.columns
        ):

            article_column = "text"
            summary_column = "headlines"


        elif (
            "clean_text" in self.df.columns
            and
            "clean_summary" in self.df.columns
        ):

            article_column = "clean_text"
            summary_column = "clean_summary"


        else:

            print(
                "\nERROR: Required columns not found."
            )

            print(
                "Expected one of these combinations:"
            )

            print(
                "1. text + headlines"
            )

            print(
                "2. clean_text + clean_summary"
            )

            raise SystemExit


        # ---------------------------------------------
        # Rename columns internally
        # ---------------------------------------------

        self.df = self.df[
            [
                article_column,
                summary_column
            ]
        ].copy()

        self.df.columns = [
            "text",
            "headlines"
        ]


        # ---------------------------------------------
        # Remove missing values
        # ---------------------------------------------

        self.df = self.df.dropna(
            subset=[
                "text",
                "headlines"
            ]
        )


        # ---------------------------------------------
        # Remove duplicates
        # ---------------------------------------------

        self.df = self.df.drop_duplicates(
            subset=[
                "text",
                "headlines"
            ]
        )


        # ---------------------------------------------
        # Convert to string
        # ---------------------------------------------

        self.df["text"] = (
            self.df["text"]
            .astype(str)
        )

        self.df["headlines"] = (
            self.df["headlines"]
            .astype(str)
        )


        # ---------------------------------------------
        # Remove empty rows
        # ---------------------------------------------

        self.df = self.df[
            self.df["text"].str.strip() != ""
        ]

        self.df = self.df[
            self.df["headlines"].str.strip() != ""
        ]


        # ---------------------------------------------
        # Add start/end tokens
        # ---------------------------------------------

        self.df["decoder_input"] = (
            "sostok "
            +
            self.df["headlines"]
        )

        self.df["decoder_target"] = (
            self.df["headlines"]
            +
            " eostok"
        )


        print(
            "\nDataset Shape After Cleaning:",
            self.df.shape
        )

        print(
            "\nSample Article:"
        )

        print(
            self.df["text"].iloc[0]
        )

        print(
            "\nSample Headline:"
        )

        print(
            self.df["headlines"].iloc[0]
        )


    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    def split_data(self):

        print("\n" + "=" * 70)
        print("Splitting Dataset...")
        print("=" * 70)


        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=0.20,
            random_state=42
        )


        print(
            "Training Samples:",
            len(self.train_df)
        )

        print(
            "Testing Samples:",
            len(self.test_df)
        )


    # =====================================================
    # TOKENIZATION
    # =====================================================

    def tokenize(self):

        print("\n" + "=" * 70)
        print("Tokenizing Dataset...")
        print("=" * 70)


        # ---------------------------------------------
        # Article tokenizer
        # ---------------------------------------------

        self.article_tokenizer.fit_on_texts(
            self.train_df["text"]
        )


        # ---------------------------------------------
        # Summary tokenizer
        # ---------------------------------------------

        self.summary_tokenizer.fit_on_texts(
            self.train_df[
                "decoder_input"
            ].tolist()
            +
            self.train_df[
                "decoder_target"
            ].tolist()
        )


        # ---------------------------------------------
        # Encoder train sequences
        # ---------------------------------------------

        encoder_train_sequences = (
            self.article_tokenizer
            .texts_to_sequences(
                self.train_df["text"]
            )
        )


        # ---------------------------------------------
        # Encoder test sequences
        # ---------------------------------------------

        encoder_test_sequences = (
            self.article_tokenizer
            .texts_to_sequences(
                self.test_df["text"]
            )
        )


        # ---------------------------------------------
        # Encoder padding
        # ---------------------------------------------

        self.encoder_train = pad_sequences(
            encoder_train_sequences,
            maxlen=self.max_article_length,
            padding="post",
            truncating="post"
        )

        self.encoder_test = pad_sequences(
            encoder_test_sequences,
            maxlen=self.max_article_length,
            padding="post",
            truncating="post"
        )


        # ---------------------------------------------
        # Decoder input train
        # ---------------------------------------------

        decoder_input_train_sequences = (
            self.summary_tokenizer
            .texts_to_sequences(
                self.train_df[
                    "decoder_input"
                ]
            )
        )


        # ---------------------------------------------
        # Decoder input test
        # ---------------------------------------------

        decoder_input_test_sequences = (
            self.summary_tokenizer
            .texts_to_sequences(
                self.test_df[
                    "decoder_input"
                ]
            )
        )


        self.decoder_input_train = pad_sequences(
            decoder_input_train_sequences,
            maxlen=self.max_summary_length,
            padding="post",
            truncating="post"
        )

        self.decoder_input_test = pad_sequences(
            decoder_input_test_sequences,
            maxlen=self.max_summary_length,
            padding="post",
            truncating="post"
        )


        # ---------------------------------------------
        # Decoder target train
        # ---------------------------------------------

        decoder_target_train_sequences = (
            self.summary_tokenizer
            .texts_to_sequences(
                self.train_df[
                    "decoder_target"
                ]
            )
        )


        # ---------------------------------------------
        # Decoder target test
        # ---------------------------------------------

        decoder_target_test_sequences = (
            self.summary_tokenizer
            .texts_to_sequences(
                self.test_df[
                    "decoder_target"
                ]
            )
        )


        self.decoder_target_train = pad_sequences(
            decoder_target_train_sequences,
            maxlen=self.max_summary_length,
            padding="post",
            truncating="post"
        )

        self.decoder_target_test = pad_sequences(
            decoder_target_test_sequences,
            maxlen=self.max_summary_length,
            padding="post",
            truncating="post"
        )


        # ---------------------------------------------
        # Expand target dimension
        # ---------------------------------------------

        self.decoder_target_train = np.expand_dims(
            self.decoder_target_train,
            axis=-1
        )

        self.decoder_target_test = np.expand_dims(
            self.decoder_target_test,
            axis=-1
        )


        # ---------------------------------------------
        # Vocabulary sizes
        # ---------------------------------------------

        self.article_vocab_size = min(
            self.max_words,
            len(
                self.article_tokenizer.word_index
            ) + 1
        )

        self.summary_vocab_size = min(
            self.max_words,
            len(
                self.summary_tokenizer.word_index
            ) + 1
        )


        print(
            "\nEncoder Train Shape:",
            self.encoder_train.shape
        )

        print(
            "Encoder Test Shape:",
            self.encoder_test.shape
        )

        print(
            "Decoder Input Train Shape:",
            self.decoder_input_train.shape
        )

        print(
            "Decoder Target Train Shape:",
            self.decoder_target_train.shape
        )

        print(
            "\nArticle Vocabulary Size:",
            self.article_vocab_size
        )

        print(
            "Summary Vocabulary Size:",
            self.summary_vocab_size
        )

        print(
            "\nTokenization Completed Successfully."
        )


    # =====================================================
    # BUILD MODEL
    # =====================================================

    def build_model(self):

        print("\n" + "=" * 70)
        print("Building Encoder-Decoder GRU Model...")
        print("=" * 70)


        # =================================================
        # Encoder Input
        # =================================================

        encoder_inputs = Input(
            shape=(
                self.max_article_length,
            ),
            name="encoder_inputs"
        )


        # =================================================
        # Encoder Embedding
        # =================================================

        encoder_embedding_layer = Embedding(
            input_dim=self.article_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="encoder_embedding"
        )


        encoder_embedding = encoder_embedding_layer(
            encoder_inputs
        )


        # =================================================
        # Encoder GRU
        # =================================================

        encoder_gru_layer = GRU(
            self.gru_units,
            return_state=True,
            name="encoder_gru"
        )


        encoder_output, encoder_state = (
            encoder_gru_layer(
                encoder_embedding
            )
        )


        # =================================================
        # Decoder Inputs
        # =================================================

        decoder_inputs = Input(
            shape=(
                self.max_summary_length,
            ),
            name="decoder_inputs"
        )


        # =================================================
        # Decoder Embedding
        # =================================================

        decoder_embedding_layer = Embedding(
            input_dim=self.summary_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="decoder_embedding"
        )


        decoder_embedding = decoder_embedding_layer(
            decoder_inputs
        )


        # =================================================
        # Decoder GRU
        # =================================================

        decoder_gru_layer = GRU(
            self.gru_units,
            return_sequences=True,
            return_state=True,
            name="decoder_gru"
        )


        decoder_outputs, decoder_state = (
            decoder_gru_layer(
                decoder_embedding,
                initial_state=encoder_state
            )
        )


        # =================================================
        # Output Dense Layer
        # =================================================

        decoder_dense_layer = Dense(
            self.summary_vocab_size,
            activation="softmax",
            name="decoder_output"
        )


        decoder_outputs = decoder_dense_layer(
            decoder_outputs
        )


        # =================================================
        # Complete Model
        # =================================================

        self.model = Model(
            inputs=[
                encoder_inputs,
                decoder_inputs
            ],
            outputs=decoder_outputs,
            name="gru_seq2seq_summarizer"
        )


        # =================================================
        # Compile
        # =================================================

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )


        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy"
            ]
        )


        print(
            "\nGRU Model Created Successfully."
        )

        self.model.summary()


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    def train(self):

        print("\n" + "=" * 70)
        print("GRU Training Started...")
        print("=" * 70)


        os.makedirs(
            "models",
            exist_ok=True
        )

        os.makedirs(
            "outputs",
            exist_ok=True
        )


        # ---------------------------------------------
        # Callbacks
        # ---------------------------------------------

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )


        model_checkpoint = ModelCheckpoint(
            filepath="models/best_gru_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )


        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )


        # ---------------------------------------------
        # Training
        # ---------------------------------------------

        self.history = self.model.fit(

            [
                self.encoder_train,
                self.decoder_input_train
            ],

            self.decoder_target_train,

            validation_data=(

                [
                    self.encoder_test,
                    self.decoder_input_test
                ],

                self.decoder_target_test

            ),

            epochs=self.epochs,

            batch_size=self.batch_size,

            callbacks=[
                early_stopping,
                model_checkpoint,
                reduce_lr
            ],

            verbose=1
        )


        print(
            "\nGRU Training Completed Successfully."
        )


    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self):

        print("\n" + "=" * 70)
        print("Evaluating GRU Model...")
        print("=" * 70)


        loss, accuracy = self.model.evaluate(

            [
                self.encoder_test,
                self.decoder_input_test
            ],

            self.decoder_target_test,

            batch_size=self.batch_size,

            verbose=1
        )


        print(
            f"\nTest Loss     : {loss:.4f}"
        )

        print(
            f"Test Accuracy : {accuracy:.4f}"
        )

        print(
            f"Accuracy (%)  : {accuracy * 100:.2f}%"
        )


        # ---------------------------------------------
        # Save evaluation
        # ---------------------------------------------

        os.makedirs(
            "outputs",
            exist_ok=True
        )


        evaluation_df = pd.DataFrame(
            {
                "Metric": [
                    "Test Loss",
                    "Test Accuracy",
                    "Accuracy Percentage"
                ],

                "Value": [
                    loss,
                    accuracy,
                    accuracy * 100
                ]
            }
        )


        evaluation_df.to_csv(
            "outputs/evaluation_results.csv",
            index=False
        )


        print(
            "\nEvaluation results saved:"
        )

        print(
            "outputs/evaluation_results.csv"
        )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    def save_model(self):

        print("\n" + "=" * 70)
        print("Saving GRU Model...")
        print("=" * 70)


        os.makedirs(
            "models",
            exist_ok=True
        )


        # ---------------------------------------------
        # Save model
        # ---------------------------------------------

        self.model.save(
            "models/gru_model.keras"
        )


        # ---------------------------------------------
        # Save article tokenizer
        # ---------------------------------------------

        with open(
            "models/article_tokenizer.pkl",
            "wb"
        ) as file:

            pickle.dump(
                self.article_tokenizer,
                file
            )


        # ---------------------------------------------
        # Save summary tokenizer
        # ---------------------------------------------

        with open(
            "models/summary_tokenizer.pkl",
            "wb"
        ) as file:

            pickle.dump(
                self.summary_tokenizer,
                file
            )


        print(
            "Model saved:"
        )

        print(
            "models/gru_model.keras"
        )

        print(
            "Tokenizers saved successfully."
        )


    # =====================================================
    # SAVE TEST DATA
    # =====================================================

    def save_test_data(self):

        os.makedirs(
            "outputs",
            exist_ok=True
        )


        self.test_df[
            [
                "text",
                "headlines"
            ]
        ].to_csv(
            "outputs/gru_test_data.csv",
            index=False
        )


        print(
            "\nTest data saved:"
        )

        print(
            "outputs/gru_test_data.csv"
        )


    # =====================================================
    # PLOT GRAPHS
    # =====================================================

    def plot_graphs(self):

        print("\n" + "=" * 70)
        print("Generating Training Graphs...")
        print("=" * 70)


        os.makedirs(
            "outputs",
            exist_ok=True
        )


        # =================================================
        # Loss graph
        # =================================================

        plt.figure(
            figsize=(8, 5)
        )


        plt.plot(
            self.history.history[
                "loss"
            ],
            label="Training Loss"
        )


        plt.plot(
            self.history.history[
                "val_loss"
            ],
            label="Validation Loss"
        )


        plt.title(
            "GRU Training and Validation Loss"
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
            "outputs/gru_loss.png",
            dpi=300,
            bbox_inches="tight"
        )


        plt.show()

        plt.close()


        # =================================================
        # Accuracy graph
        # =================================================

        plt.figure(
            figsize=(8, 5)
        )


        plt.plot(
            self.history.history[
                "accuracy"
            ],
            label="Training Accuracy"
        )


        plt.plot(
            self.history.history[
                "val_accuracy"
            ],
            label="Validation Accuracy"
        )


        plt.title(
            "GRU Training and Validation Accuracy"
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
            "outputs/gru_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )


        plt.show()

        plt.close()


        print(
            "\nGraphs saved successfully:"
        )

        print(
            "outputs/gru_loss.png"
        )

        print(
            "outputs/gru_accuracy.png"
        )


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    gru = GRUSummarizer()


    # =====================================================
    # CORRECT DATASET PATH
    # Screenshot එක අනුව folder එක dataset/
    # =====================================================

    dataset_path = (
        "dataset/clean_news_summary.csv"
    )


    gru.load_dataset(
        dataset_path
    )


    gru.split_data()


    gru.tokenize()


    gru.build_model()


    gru.train()


    gru.evaluate()


    gru.save_model()


    gru.save_test_data()


    gru.plot_graphs()