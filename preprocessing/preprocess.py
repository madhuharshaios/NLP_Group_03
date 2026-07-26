import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clean import clean_text


def preprocess_dataset():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = pd.read_csv("data/news_summary.csv", encoding="latin-1")

    print("\nOriginal Shape :", df.shape)

    # Required columns only
    df = df[["text", "headlines"]]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("After Cleaning :", df.shape)

    print("\nCleaning Articles...")
    df["text"] = df["text"].apply(clean_text)

    print("Cleaning Headlines...")
    df["headlines"] = df["headlines"].apply(clean_text)

    df.to_csv("data/clean_news_summary.csv", index=False)

    print("\nDataset Saved Successfully")
    print("Output : data/clean_news_summary.csv")


if __name__ == "__main__":
    preprocess_dataset()