import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
MODEL_PATH = MODEL_DIR / "hate_speech_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "hate_speech_tokenizer.json"
CURVE_PATH = FIGURES_DIR / "training_curves.png"

VOCAB_SIZE = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 8


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_tokenizer(texts):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def build_model() -> Sequential:
    model = Sequential(
        [
            Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
            LSTM(LSTM_UNITS),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)

    tokenizer = build_tokenizer(train_df["clean_text"].astype(str).tolist())
    train_sequences = tokenizer.texts_to_sequences(train_df["clean_text"].astype(str).tolist())
    val_sequences = tokenizer.texts_to_sequences(val_df["clean_text"].astype(str).tolist())

    x_train = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")
    x_val = pad_sequences(val_sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")
    y_train = train_df["label"].astype(int).values
    y_val = val_df["label"].astype(int).values

    model = build_model()
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    model.save(MODEL_PATH)

    with TOKENIZER_PATH.open("w", encoding="utf-8") as file_handle:
        file_handle.write(tokenizer.to_json())

    save_training_curves(history)

    summary = {
        "vocab_size": VOCAB_SIZE,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "curve_path": str(CURVE_PATH),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    summary_path = MODEL_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()