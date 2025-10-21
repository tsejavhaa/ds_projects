import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

current_dir = os.path.dirname(os.path.abspath(__file__))
# Load model and vectorizer
model = joblib.load(os.path.join(current_dir, "logistic_regression_model.pkl"))
tfidf = joblib.load(os.path.join(current_dir, "tfidf_vectorizer.pkl"))

# UI
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline below to detect if it's **real or fake**.")

st.write("*Example input:*")
st.write("- The government announced a new policy to increase education funding.")
st.write("- Breaking: Alien spaceship lands in the middle of the desert!")

text_input = st.text_area("Enter news text:", height=200)

if st.button("Predict"):
    if len(text_input.strip()) == 0:
        st.warning("Please enter a news article or headline.")
    else:
        vectorized = tfidf.transform([text_input])
        pred_prob = model.predict_proba(vectorized)[0][1]
        label = "✅ REAL News" if pred_prob >= 0.5 else "🚫 FAKE News"

        st.subheader(label)
        st.metric("Confidence", f"{pred_prob*100:.2f}%")

        if pred_prob >= 0.5:
            st.info("This article seems legitimate, but fact-checking is still advised.")
        else:
            st.error("This article may contain misinformation.")