import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from rouge_score import rouge_scorer



# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    r"D:\NLP_Member 1 news summerize\data\clean_news.csv"
)


articles = df["article"].astype(str)
summaries = df["summary"].astype(str)



# =========================
# Load CNN Model
# =========================

model = load_model(
    r"D:\NLP_Member 1 news summerize\models\cnn_model.keras"
)


print("CNN Model Loaded Successfully")



# =========================
# Tokenizer
# =========================

tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)


tokenizer.fit_on_texts(articles)



# =========================
# Generate CNN Predictions
# =========================


generated_summaries = []


for article in articles[:100]:

    sequence = tokenizer.texts_to_sequences(
        [article]
    )


    padded = pad_sequences(
        sequence,
        maxlen=200,
        padding="post"
    )


    prediction = model.predict(
        padded,
        verbose=0
    )


    # CNN confidence output
    summary = (
        "Generated summary confidence: "
        + str(round(float(prediction[0][0]),3))
    )


    generated_summaries.append(summary)



# Real summaries

actual_summaries = summaries[:100].tolist()



# =========================
# ROUGE Evaluation
# =========================

scorer = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL"
    ],
    use_stemmer=True
)


rouge1 = []
rouge2 = []
rougeL = []


for ref, pred in zip(
    actual_summaries,
    generated_summaries
):

    score = scorer.score(
        ref,
        pred
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



print("\nCNN ROUGE Results")

print(
    "ROUGE-1:",
    np.mean(rouge1)
)

print(
    "ROUGE-2:",
    np.mean(rouge2)
)

print(
    "ROUGE-L:",
    np.mean(rougeL)
)



# =========================
# Save Results
# =========================


results = pd.DataFrame({

    "actual_summary": actual_summaries,

    "generated_summary": generated_summaries

})


results.to_csv(
    r"D:\NLP_Member 1 news summerize\data\cnn_results.csv",
    index=False
)


print("\nCNN results saved successfully!")