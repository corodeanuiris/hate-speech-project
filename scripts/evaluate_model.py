import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "hate_speech_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "hate_speech_tokenizer.json"
REPORT_PATH = MODEL_DIR / "evaluation_report.json"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"

MAX_LENGTH = 100


def load_tokenizer(path: Path):
    tokenizer_json = path.read_text(encoding="utf-8")
    return tokenizer_from_json(tokenizer_json)


def plot_confusion_matrix(matrix):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Neutral", "Toxic"])
    plt.yticks(tick_marks, ["Neutral", "Toxic"])

    threshold = matrix.max() / 2.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            plt.text(
                col_index,
                row_index,
                format(matrix[row_index, col_index], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()


def main() -> None:
    test_df = pd.read_csv(TEST_PATH)
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    sequences = tokenizer.texts_to_sequences(test_df["clean_text"].astype(str).tolist())
    x_test = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")
    y_true = test_df["label"].astype(int).values

    probabilities = model.predict(x_test, verbose=0).ravel()
    y_pred = (probabilities >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(matrix)

    summary = {
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "test_rows": int(len(test_df)),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
        "classification_report": report,
    }

    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()