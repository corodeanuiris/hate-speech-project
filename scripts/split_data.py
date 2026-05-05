import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/cleaned_dataset.csv") # loads the cleaned CSV, keeping only clean_text and label

df_model = df[["clean_text", "label"]].copy()

train_df, temp_df = train_test_split(  # we split into train (70%), validation (15%), and test (15%)
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

# it takes the balanced data file and splits it into train/validation/test
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# 75% train => the model learns from this data (sees examples and adjusts its weights during training)
# 15% validation => used DURING training, but the model does NOT train on this. used to tune parameters (learning rate, model size etcc)
# 15% test => touched ONLY ONCE, after all decisions are made, to get an honest estimate of real-world performance