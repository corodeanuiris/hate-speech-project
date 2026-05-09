import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
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
THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.json"

VOCAB_SIZE = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 30


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_tokenizer(texts):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def calibrate_threshold(model, x_val, y_val) -> float:
    """Find the decision threshold that maximises macro-F1 on the validation set.

    We sweep every candidate threshold returned by precision_recall_curve so the
    search is exact (no grid step needed) and guaranteed to include the optimum.
    """
    val_probs = model.predict(x_val, verbose=0).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # thresholds has one fewer element than precisions/recalls — align them
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in thresholds:
        preds = (val_probs >= threshold).astype(int)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def build_model() -> Sequential:
    model = Sequential(
        [
            Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
            SpatialDropout1D(0.2),          # drops entire feature maps, safe on CPU
            Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
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
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

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

    # Calibrate decision threshold on the validation set
    optimal_threshold = calibrate_threshold(model, x_val, y_val)
    THRESHOLD_PATH.write_text(
        json.dumps({"optimal_threshold": optimal_threshold}, indent=2), encoding="utf-8"
    )

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
        "optimal_threshold": optimal_threshold,
        "threshold_path": str(THRESHOLD_PATH),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    summary_path = MODEL_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()