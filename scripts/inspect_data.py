import pandas as pd

df = pd.read_csv("data/raw/raw_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

if "class" in df.columns:
    print("\nClass distribution:")
    print(df["class"].value_counts())
else:
    print("\nNo 'class' column found. Please check your dataset.")