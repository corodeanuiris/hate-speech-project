import pandas as pd
import re

df = pd.read_csv("data/raw/raw_dataset.csv") # we load raw data from CSV 

print("Original shape:", df.shape)
print("Columns found:", df.columns.tolist())

df = df[["tweet", "class"]].copy()  # we keep only the tweet and class columns
df = df.dropna()

def clean_text(text):     # cleaning the raw data by removing urls, mentions, hashtags, punctuation, and extra whitespace
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["tweet"].apply(clean_text) 
df = df[df["clean_text"] != ""]

df["label"] = df["class"].apply(lambda x: 0 if x == 2 else 1) # we map the data: 0 for non-hate speech (class 2) and 1 for hate speech (classes 0 and 1)

# Balance classes by undersampling the majority class (toxic) to match the minority class (neutral)
# This prevents the model from predicting everything as toxic due to skewed training data 

# in this file the data is BALANCED => cleaned, labeled, BALANCED (discards toxic tweets until both classes are equal) and saves
minority_size = df["label"].value_counts().min()
df_majority = df[df["label"] == 1].sample(minority_size, random_state=42)
df_minority = df[df["label"] == 0]
df_balanced = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("\nCleaned shape (before balancing):", df.shape)
print("\nBalanced shape:", df_balanced.shape)
print("\nBinary label distribution (balanced):")
print(df_balanced["label"].value_counts())
print("\nDone. File saved as data/processed/cleaned_dataset.csv")