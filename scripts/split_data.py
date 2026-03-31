import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/cleaned_dataset.csv")

df_model = df[["clean_text", "label"]].copy()

train_df, temp_df = train_test_split(
    df_model,
    test_size=0.30,
    random_state=42,
    stratify=df_model["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)