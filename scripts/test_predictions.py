"""Interactive hate speech classifier.

Run from the hate-speech-project/ directory:
    python3 scripts/test_predictions.py

Type a sentence and press Enter to classify it.
Type 'quit' or 'exit' to stop.
"""

import json
import re
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hate_speech_lstm.keras"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "hate_speech_tokenizer.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "optimal_threshold.json"

MAX_LENGTH = 100


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_threshold(path: Path) -> float:
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["optimal_threshold"])
    return 0.5


def main() -> None:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    threshold = load_threshold(THRESHOLD_PATH)
    print(f"Model ready. Decision threshold: {threshold:.4f}")
    print("=" * 50)
    print("Type a sentence to classify. Type 'quit' to exit.")
    print("=" * 50)

    while True:
        try:
            text = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text:
            continue

        if text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        cleaned = clean_text(text)
        if not cleaned:
            print("Could not process that input (no recognisable words).")
            continue

        seq = tokenizer.texts_to_sequences([cleaned])
        x = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
        prob = float(model.predict(x, verbose=0)[0][0])

        label = "TOXIC" if prob >= threshold else "NEUTRAL"
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"  Result    : {label}")
        print(f"  Confidence: [{bar}] {prob:.3f}")


if __name__ == "__main__":
    main()
