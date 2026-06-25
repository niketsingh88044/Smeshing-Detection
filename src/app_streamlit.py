import streamlit as st
import requests

st.set_page_config(page_title="Smishing Detector", page_icon="📱")
st.title("📱 Smishing (SMS Phishing) Detection Demo")

st.write("Enter any SMS message below and see if it's safe or smishing/spam.")

sms_text = st.text_area("Enter SMS message:")

if st.button("Analyze"):
    if not sms_text.strip():
        st.warning("Please enter a message to analyze.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": sms_text}
            )
            result = response.json()
            label = result["label"]
            prob_spam = result["probabilities"]["spam"]

            if label == "Smishing/Spam":
                st.error(f"🚨 Bhaiya ji spam h (Probability: {prob_spam:.2f})")
            else:
                st.success(f"✅ safe to use (Probability of spam: {prob_spam:.2f})")

        except Exception as e:
            st.error(f"⚠️ API error: {e}")
