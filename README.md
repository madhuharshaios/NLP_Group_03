# Text Summarization System Using Machine Learning and Deep Learning

## Group Information

**Group Number:** NLP_Group_03

### Group Members

| Student ID | Member | ML Model | DL Model |
|------------|---------|----------|----------|
| CIT-24-01-0301 | Member 01 | TextRank | CNN |
| CIT-24-01-0070 | Member 02 | LSA | LSTM |
| CIT-24-01-0082 | Member 03 | TF-IDF Sentence Ranking | GRU |

---

## Problem Statement

Reading lengthy news articles can be time-consuming for users. This project aims to develop an automated Text Summarization System using Natural Language Processing (NLP) techniques. The system generates concise and meaningful summaries from long news articles while preserving the most important information.

The project compares multiple Machine Learning and Deep Learning models to identify the most effective approach for text summarization.

---

## Dataset Information

**Dataset Name:** News Summary Dataset

**Dataset Source:** Kaggle

**Dataset URL:**
https://www.kaggle.com/datasets/sunnysai12345/news-summary

**Task Type:** Text Summarization

**Input:**
- News Article Text

**Output:**
- Generated Summary

**Language:**
- English

---

## Project Structure

```text
NLP_Group_03/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── screenshots/
├── videos/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/madhuharshaios/NLP_Group_03.git
cd NLP_Group_03
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### 3. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Run Model Training

```bash
python src/member1_textRank.py
python src/member2_lsa.py
python src/member3_tfidf.py
```

### Run Deep Learning Models

```bash
python src/member1_cnn.py
python src/member2_lstm.py
python src/member3_gru.py
```

### Run Web Application

```bash
streamlit run app.py
```

---
## 🧠 Model Summary

### 👤 Member 01 (CIT-24-01-0301)

**Machine Learning Model:**
- TextRank

**Deep Learning Model:**
- CNN (Convolutional Neural Network)

**Purpose:**
- TextRank is used for extractive summarization by ranking important sentences.
- CNN captures local text patterns and improves feature-based summarization.

---

### 👤 Member 02 (CIT-24-01-0070)

**Machine Learning Model:**
- LSA (Latent Semantic Analysis)

**Deep Learning Model:**
- LSTM (Long Short-Term Memory)

**Purpose:**
- LSA extracts hidden semantic topics from text.
- LSTM learns long-term dependencies in sequential text for better summarization.

---

### 👤 Member 03 (CIT-24-01-0082)

**Machine Learning Model:**
- TF-IDF Sentence Ranking

**Deep Learning Model:**
- GRU (Gated Recurrent Unit)

**Purpose:**
- TF-IDF ranks sentences based on keyword importance.
- GRU improves sequence learning with efficient recurrent processing.

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROUGE Score

---

## Results Summary

The project compares the performance of six models:

| Model                   | Type | Status    |
| ----------------------- | ---- | --------- |
| TextRank                | ML   | Evaluated |
| LSA                     | ML   | Evaluated |
| TF-IDF Sentence Ranking | ML   | Evaluated |
| CNN                     | DL   | Evaluated |
| LSTM                    | DL   | Evaluated |
| GRU                     | DL   | Evaluated |


The best-performing model will be selected and integrated into the final Streamlit web application for text summarization.

---

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow / Keras
- NLTK
- Streamlit
- Git & GitHub

---

## Ethical Considerations

- Dataset bias and fairness
- Potential misinformation in generated summaries
- Privacy considerations
- Responsible AI usage
- Human validation of generated outputs

---

## Repository

GitHub Repository:

https://github.com/madhuharshaios/NLP_Group_03
