import os
import re

import joblib
import numpy as np
import streamlit as st
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PCA_PATH = os.path.join(BASE_DIR, "pca.pkl")
EMBED_MODEL_PATH = os.path.join(BASE_DIR, "embed_model.txt")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.txt")
GLOVE_BINARY_PATH = os.path.join(BASE_DIR, "glove.kv")


@st.cache_resource
def load_artifacts():
    with open(EMBED_MODEL_PATH) as f:
        embed_model_name = f.read().strip()
    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())
    if os.path.exists(GLOVE_BINARY_PATH):
        kv = KeyedVectors.load(GLOVE_BINARY_PATH, mmap="r")
    else:
        kv = gensim_api.load(embed_model_name)
        kv.save(GLOVE_BINARY_PATH)
    pca = joblib.load(PCA_PATH)
    model = joblib.load(MODEL_PATH)
    return kv, pca, model, threshold


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def embed_message(text: str, kv) -> np.ndarray:
    tokens = simple_preprocess(text)
    vectors = [kv[t] for t in tokens if t in kv]
    if not vectors:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


st.set_page_config(page_title="Smishing Detector", page_icon="📱")
st.title("📱 Smishing (SMS Phishing) Detection Demo")
st.write("Enter any SMS message below and see if it's safe or smishing/spam.")

with st.spinner("Loading model (first time may take a moment)..."):
    kv, pca, model, threshold = load_artifacts()

sms_text = st.text_area("Enter SMS message:")

if st.button("Analyze"):
    if not sms_text.strip():
        st.warning("Please enter a message to analyze.")
    else:
        msg_clean = clean_text(sms_text)
        embedding = embed_message(msg_clean, kv).reshape(1, -1)
        reduced = pca.transform(embedding)

        probs = model.predict_proba(reduced)[0]
        spam_idx = list(model.classes_).index("spam")
        prob_spam = float(probs[spam_idx])
        is_spam = prob_spam >= threshold

        if is_spam:
            st.error(f"🚨 Bhaiya ji spam h (Probability: {prob_spam:.2f})")
        else:
            st.success(f"✅ safe to use (Probability of spam: {prob_spam:.2f})")
