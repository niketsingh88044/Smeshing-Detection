import os
import re
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(page_title="Smishing Detector", page_icon="📱")
st.title("📱 Smishing (SMS Phishing) Detection Demo")
st.write("Enter any SMS message below and see if it's safe or smishing/spam.")

model, vectorizer = load_artifacts()

sms_text = st.text_area("Enter SMS message:")

if st.button("Analyze"):
    if not sms_text.strip():
        st.warning("Please enter a message to analyze.")
    else:
        msg_clean = clean_text(sms_text)
        msg_vec = vectorizer.transform([msg_clean])

        pred = model.predict(msg_vec)[0]
        probs = model.predict_proba(msg_vec)[0]
        prob_spam = float(probs[1])

        if str(pred) == "spam":
            st.error(f"🚨 Bhaiya ji spam h (Probability: {prob_spam:.2f})")
        else:
            st.success(f"✅ safe to use (Probability of spam: {prob_spam:.2f})")
