# News Summarization using CNN

## Project Overview

This project implements a news summarization system using a Convolutional Neural Network (CNN). The model is trained on a news summarization dataset to learn patterns between news articles and their summaries. The trained model is evaluated using ROUGE metrics and can be used through a Streamlit web application.

---

## Project Structure

```
member_01_cnn/
│
├── app/
│   └── app.py
│
├── data/
│   ├── clean_news.csv
│   ├── news_summary.csv
│   └── cnn_predictions.csv
│
├── evaluation/
│   ├── evaluate_cnn.py
│   └── compare_models.py
│
├── models/
│   ├── cnn_model.py
│   └── cnn_model.keras
│
├── notebooks/
│   ├── load_data.py
│   ├── 03_preprocessing.py
│   └── 04_eda.py
│
├── reports/
├── screenshots/
├── requirements.txt
└── README.md
```

---

## Features

- Load and preprocess news dataset
- CNN model training
- Generate summaries
- Evaluate using ROUGE metrics
- Compare model performance
- Streamlit web application

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Project

Load dataset

```bash
python notebooks/load_data.py
```

Preprocess dataset

```bash
python notebooks/03_preprocessing.py
```

Train CNN model

```bash
python models/cnn_model.py
```

Evaluate CNN

```bash
python evaluation/evaluate_cnn.py
```

Run Streamlit App

```bash
streamlit run app/app.py
```

---

## Author

Yuradi pramuditha - Member 01