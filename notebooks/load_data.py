import pandas as pd

# Dataset path
path = r"D:\NLP_Member 1 news summerize\data\news_summary.csv"

# Read dataset
df = pd.read_csv(path, encoding="latin-1")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())