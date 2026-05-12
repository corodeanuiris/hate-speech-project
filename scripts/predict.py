"""Interactive hate speech classifier.

Run from the hate-speech-project/ directory:
    python3 scripts/predict.py

Type a sentence and press Enter to classify it.
Type 'quit' or 'exit' to stop.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
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


def load_calibration(path: Path) -> tuple:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["optimal_threshold"]), float(data.get("temperature", 1.0))
    return 0.5, 1.0


def apply_temperature(prob: float, temperature: float) -> float:
    if temperature == 1.0:
        return prob
    prob = max(1e-7, min(1 - 1e-7, prob))
    return float(expit(logit(prob) / temperature))


def main() -> None:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    threshold, temperature = load_calibration(THRESHOLD_PATH)
    print(f"Model ready. Threshold: {threshold:.4f}  Temperature: {temperature:.4f}")
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
        raw_prob = float(model.predict(x, verbose=0)[0][0])
        prob = apply_temperature(raw_prob, temperature)

        label = "TOXIC" if prob >= threshold else "NEUTRAL"
        # Show confidence in the predicted label, not raw toxic probability
        confidence = prob if label == "TOXIC" else (1.0 - prob)
        bar_len = int(confidence * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"  Result    : {label}")
        print(f"  Confidence: [{bar}] {confidence:.3f}")


if __name__ == "__main__":
    main()
