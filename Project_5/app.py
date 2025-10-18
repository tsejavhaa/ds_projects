from debugpy.common.timestamp import current
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load pre-trained model and vectorizer
current_dir = os.path.dirname(os.path.abspath('__file__')) 
model_path = os.path.join(current_dir, 'Project_5/sentiment_model.pkl')
vectorizer_path = os.path.join(current_dir, 'Project_5/tfidf_vectorizer.pkl')
model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)

st.title("🛍️ Sentiment Analysis on Product Reviews")
st.write("Predict whether a customer review is Positive or Negative.")

review = st.text_area("Enter a product review:")

if st.button("Analyze"):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    st.subheader("Prediction Result:")
    st.success("😊 Positive Review") if prediction == 1 else st.error("😞 Negative Review")