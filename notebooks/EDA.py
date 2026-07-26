import os
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud


# -------------------------------------------------
# 1. Project root path එක හඳුනාගැනීම
# -------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)
NOTEBOOKS_FOLDER = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(NOTEBOOKS_FOLDER)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "clean_news_summary.csv"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "eda"
)

# outputs/eda folder එක නැත්නම් create කරනවා
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -------------------------------------------------
# 2. Dataset එක load කිරීම
# -------------------------------------------------

print("=" * 60)
print("Loading Clean Dataset")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print("\nError: clean_news_summary.csv file එක හමු නොවුණා.")
    print("මුලින් මේ command එක run කරන්න:")
    print("python preprocessing/preprocess.py")
    raise SystemExit


print("\nDataset loaded successfully.")


# -------------------------------------------------
# 3. Dataset overview
# -------------------------------------------------

print("\n" + "=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nDataset Information")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nFirst 5 Rows")
print(df.head())


# -------------------------------------------------
# 4. Missing values handle කිරීම
# -------------------------------------------------

required_columns = ["text", "headlines"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nError: Required columns missing:", missing_columns)
    raise SystemExit

df = df.dropna(subset=required_columns)
df = df.drop_duplicates(subset=required_columns)

df["text"] = df["text"].astype(str)
df["headlines"] = df["headlines"].astype(str)


# -------------------------------------------------
# 5. Article සහ headline lengths
# -------------------------------------------------

df["article_length"] = df["text"].apply(
    lambda text: len(text.split())
)

df["headline_length"] = df["headlines"].apply(
    lambda headline: len(headline.split())
)

print("\n" + "=" * 60)
print("Article and Headline Length Statistics")
print("=" * 60)

print(
    df[
        ["article_length", "headline_length"]
    ].describe()
)


# -------------------------------------------------
# 6. Article length histogram
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df["article_length"],
    bins=30,
    edgecolor="black"
)

plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()

article_histogram_path = os.path.join(
    OUTPUT_FOLDER,
    "article_length_distribution.png"
)

plt.savefig(
    article_histogram_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nSaved:", article_histogram_path)


# -------------------------------------------------
# 7. Headline length histogram
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df["headline_length"],
    bins=20,
    edgecolor="black"
)

plt.title("Headline Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()

headline_histogram_path = os.path.join(
    OUTPUT_FOLDER,
    "headline_length_distribution.png"
)

plt.savefig(
    headline_histogram_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Saved:", headline_histogram_path)


# -------------------------------------------------
# 8. Word cloud
# -------------------------------------------------

all_article_text = " ".join(df["text"].tolist())

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=200
).generate(all_article_text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of News Articles")
plt.tight_layout()

wordcloud_path = os.path.join(
    OUTPUT_FOLDER,
    "article_wordcloud.png"
)

plt.savefig(
    wordcloud_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Saved:", wordcloud_path)


# -------------------------------------------------
# 9. Top 20 frequent words
# -------------------------------------------------

all_words = all_article_text.split()

word_counter = Counter(all_words)

top_20_words = word_counter.most_common(20)

print("\n" + "=" * 60)
print("Top 20 Most Frequent Words")
print("=" * 60)

for word, count in top_20_words:
    print(f"{word}: {count}")


# -------------------------------------------------
# 10. Top 20 words bar chart
# -------------------------------------------------

words = [
    word for word, count in top_20_words
]

counts = [
    count for word, count in top_20_words
]

plt.figure(figsize=(12, 6))

plt.bar(words, counts)

plt.title("Top 20 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()

top_words_chart_path = os.path.join(
    OUTPUT_FOLDER,
    "top_20_words.png"
)

plt.savefig(
    top_words_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Saved:", top_words_chart_path)


# -------------------------------------------------
# 11. Length statistics save කිරීම
# -------------------------------------------------

statistics_path = os.path.join(
    OUTPUT_FOLDER,
    "length_statistics.csv"
)

df[
    ["article_length", "headline_length"]
].describe().to_csv(statistics_path)

print("Saved:", statistics_path)


# -------------------------------------------------
# 12. Final message
# -------------------------------------------------

print("\n" + "=" * 60)
print("EDA Completed Successfully")
print("=" * 60)

print("\nGenerated files are available in:")
print(OUTPUT_FOLDER)