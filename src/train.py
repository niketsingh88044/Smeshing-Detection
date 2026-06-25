import re

import joblib
import numpy as np
import pandas as pd
import gensim.downloader as gensim_api
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/SMSSpamCollection.txt"
EMBED_MODEL = "glove-wiki-gigaword-100"
EMBED_DIM = 100
PCA_COMPONENTS = 30
SPAM_THRESHOLD = 0.35
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data():
    df = pd.read_csv(
        DATA_PATH,
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


def embed_messages(texts, kv) -> np.ndarray:
    return np.vstack([embed_message(t, kv) for t in texts])


def main():
    print("Training script started")

    print(f"Loading data from {DATA_PATH}...")
    df = load_data()
    counts = df["label"].value_counts().to_dict()
    print(f"  Clean rows: {len(df)}  |  {counts}")

    print(f"\nLoading word vectors: {EMBED_MODEL}")
    print("  (first run downloads ~130MB, then cached in ~/gensim-data/)")
    kv = gensim_api.load(EMBED_MODEL)
    print(f"  Vocab size: {len(kv):,}  |  Vector dim: {kv.vector_size}")
    kv.save("glove.kv")
    print("  Cached binary: glove.kv (fast mmap load for app/evaluate)")

    print("\nEmbedding messages (mean-pooling word vectors)...")
    X = embed_messages(df["message"].tolist(), kv)
    y = df["label"].values
    print(f"  Embedding shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\nApplying SMOTE to balance training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    bal_counts = dict(zip(*np.unique(y_train_bal, return_counts=True)))
    print(f"  After SMOTE: {bal_counts}")

    print(f"\nFitting PCA ({PCA_COMPONENTS} components)...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_red = pca.fit_transform(X_train_bal)
    X_test_red = pca.transform(X_test)
    print(f"  Explained variance retained: {pca.explained_variance_ratio_.sum():.3f}")

    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_red, y_train_bal)

    spam_idx = list(rf.classes_).index("spam")
    proba_spam = rf.predict_proba(X_test_red)[:, spam_idx]

    print("\n=== Default threshold (0.50) ===")
    y_pred_default = rf.predict(X_test_red)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
    print(classification_report(y_test, y_pred_default))
    print("Confusion matrix (rows=true, cols=pred | order=ham,spam):")
    print(confusion_matrix(y_test, y_pred_default, labels=["ham", "spam"]))

    print(f"\n=== Tuned threshold ({SPAM_THRESHOLD:.2f}) ===")
    y_pred_tuned = np.where(proba_spam >= SPAM_THRESHOLD, "spam", "ham")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
    print(classification_report(y_test, y_pred_tuned))
    print("Confusion matrix (rows=true, cols=pred | order=ham,spam):")
    print(confusion_matrix(y_test, y_pred_tuned, labels=["ham", "spam"]))

    print("\nSaving artifacts...")
    joblib.dump(pca, "pca.pkl")
    joblib.dump(rf, "model.pkl")
    with open("embed_model.txt", "w") as f:
        f.write(EMBED_MODEL)
    with open("threshold.txt", "w") as f:
        f.write(str(SPAM_THRESHOLD))
    print("  Saved: model.pkl, pca.pkl, embed_model.txt, threshold.txt")
    print("Training script completed")


if __name__ == "__main__":
    main()
