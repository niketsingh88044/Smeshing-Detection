import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

GLOVE_BINARY_PATH = "glove.kv"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        on_bad_lines="skip",
        quoting=3,
        engine="python",
    )
    df = df.dropna()
    df = df[df["label"].isin(["ham", "spam"])]
    df["message"] = df["message"].astype(str).map(clean_text)
    df = df[df["message"].str.len() > 0]
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    return df


def embed_message(text: str, kv) -> np.ndarray:
    tokens = simple_preprocess(text)
    vectors = [kv[t] for t in tokens if t in kv]
    if not vectors:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def evaluate(test_path: str, output_path: str):
    with open("embed_model.txt") as f:
        embed_model_name = f.read().strip()
    with open("threshold.txt") as f:
        threshold = float(f.read().strip())
    if os.path.exists(GLOVE_BINARY_PATH):
        kv = KeyedVectors.load(GLOVE_BINARY_PATH, mmap="r")
    else:
        kv = gensim_api.load(embed_model_name)
        kv.save(GLOVE_BINARY_PATH)
    pca = joblib.load("pca.pkl")
    model = joblib.load("model.pkl")

    df = load_test_data(test_path)
    X = np.vstack([embed_message(t, kv) for t in df["message"]])
    X_red = pca.transform(X)
    y_true = df["label"].values

    spam_idx = list(model.classes_).index("spam")
    proba_spam = model.predict_proba(X_red)[:, spam_idx]
    y_pred = np.where(proba_spam >= threshold, "spam", "ham")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).tolist()

    results = {
        "accuracy": acc,
        "threshold": threshold,
        "report": report,
        "confusion_matrix": cm,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation complete. Accuracy: {acc:.4f} @ threshold {threshold}. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/SMSSpamCollection.txt")
    parser.add_argument("--output_path", default="eval.json")
    args = parser.parse_args()
    evaluate(args.test_path, args.output_path)
