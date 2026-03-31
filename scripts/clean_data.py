import pandas as pd
import re

df = pd.read_csv("data/raw/raw_dataset.csv")

print("Original shape:", df.shape)
print("Columns found:", df.columns.tolist())

df = df[["tweet", "class"]].copy()
df = df.dropna()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["tweet"].apply(clean_text)
df = df[df["clean_text"] != ""]

df["label"] = df["class"].apply(lambda x: 0 if x == 2 else 1)

df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("\nCleaned shape:", df.shape)
print("\nBinary label distribution:")
print(df["label"].value_counts())
print("\nDone. File saved as data/processed/cleaned_dataset.csv")