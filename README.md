# News Summarization using TextRank

## Project Overview

This project implements an extractive news summarization system using the TextRank algorithm. It reads news articles, preprocesses the text, generates summaries using TextRank, and evaluates the generated summaries using ROUGE metrics.

## Project Structure

```
member_01_textrank/
│
├── data/
│   ├── clean_news.csv
│   ├── news_summary.csv
│   ├── textrank_results.csv
│   ├── article_length.png
│   ├── summary_length.png
│   └── wordcloud.png
│
├── models/
│   └── textrank.py
│
├── notebooks/
│   ├── load_data.py
│   ├── 03_preprocessing.py
│   └── 04_eda.py
│
├── evaluation/
│   └── evaluate_textrank.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## Features

- Load news dataset
- Text preprocessing
- Exploratory Data Analysis (EDA)
- TextRank summarization
- ROUGE evaluation
- Streamlit web application

## Installation

```bash
pip install -r requirements.txt
```

## Run Project

Load dataset

```bash
python notebooks/load_data.py
```

Preprocess data

```bash
python notebooks/03_preprocessing.py
```

Run TextRank

```bash
python models/textrank.py
```

Evaluate

```bash
python evaluation/evaluate_textrank.py
```

Run Streamlit App

```bash
streamlit run app/app.py
```

## Author

Yuradi Pramuditha - Member 01