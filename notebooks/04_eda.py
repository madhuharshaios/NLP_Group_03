import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv(r"D:\NLP_Member 1 news summerize\data\clean_news.csv")

print("Dataset Shape:", df.shape)

# Article length
df["article_length"] = df["article"].apply(lambda x: len(str(x).split()))

# Summary length
df["summary_length"] = df["summary"].apply(lambda x: len(str(x).split()))

print("\nAverage Article Length:", round(df["article_length"].mean(), 2))
print("Average Summary Length:", round(df["summary_length"].mean(), 2))

# -----------------------------
# Article Length Histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["article_length"], bins=30)
plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")
plt.savefig(r"D:\NLP_Member 1 news summerize\data\article_length.png")
plt.show()

# -----------------------------
# Summary Length Histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["summary_length"], bins=30)
plt.title("Summary Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Summaries")
plt.savefig(r"D:\NLP_Member 1 news summerize\data\summary_length.png")
plt.show()

# -----------------------------
# Word Cloud
# -----------------------------
text = " ".join(df["article"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.savefig(r"D:\NLP_Member 1 news summerize\data\wordcloud.png")
plt.show()