import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv(
    r"D:\NLP_Member 1 news summerize\data\clean_news.csv"
)


print("Dataset Loaded")
print(df.head())


# Articles and summaries

articles = df["article"].astype(str)
summaries = df["summary"].astype(str)



# ==============================
# 2. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    articles,
    summaries,
    test_size=0.2,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



# ==============================
# 3. Tokenization
# ==============================

vocab_size = 10000
max_length = 200


tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)


tokenizer.fit_on_texts(X_train)



# Convert article text to numbers

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)



# Padding

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length,
    padding="post"
)


X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length,
    padding="post"
)


print("\nInput training shape:", X_train_pad.shape)
print("Input testing shape:", X_test_pad.shape)



# ==============================
# 4. Create CNN Labels
# ==============================
# 
# CNN needs numeric labels.
# 1 = article has summary
#


y_train_label = np.ones(len(y_train))

y_test_label = np.ones(len(y_test))


print("\nLabels created")



# ==============================
# 5. CNN Model
# ==============================


model = Sequential()


# Embedding Layer

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_length
    )
)



# Convolution Layer

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )
)



# Pooling

model.add(
    GlobalMaxPooling1D()
)



# Dropout

model.add(
    Dropout(0.5)
)



# Dense Layer

model.add(
    Dense(
        64,
        activation="relu"
    )
)



# Output Layer

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



# Build model

model.build(
    input_shape=(None, max_length)
)



# Compile

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



# Model Summary

model.summary()



# ==============================
# 6. Train CNN
# ==============================


history = model.fit(
    X_train_pad,
    y_train_label,
    epochs=10,
    batch_size=32,
    validation_data=(
        X_test_pad,
        y_test_label
    )
)



# ==============================
# 7. Save Model
# ==============================


model.save(
    r"D:\NLP_Member 1 news summerize\models\cnn_model.keras"
)


print("\nCNN Training Completed!")
print("Model saved successfully!")
