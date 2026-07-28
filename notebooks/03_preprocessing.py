import pandas as pd
import re

# Load dataset
path = r"D:\NLP_Member 1 news summerize\data\news_summary.csv"
df = pd.read_csv(path, encoding="latin-1")

# Keep only the required columns
df = df[['ctext', 'text']]

# Rename columns
df.columns = ['article', 'summary']

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()                  # lowercase
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)   # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# Apply cleaning
df["article"] = df["article"].apply(clean_text)
df["summary"] = df["summary"].apply(clean_text)

# Save cleaned dataset
df.to_csv(r"D:\NLP_Member 1 news summerize\data\clean_news.csv", index=False)

print("Preprocessing completed successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)