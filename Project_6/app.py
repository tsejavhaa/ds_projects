import streamlit as st
import pandas as pd
import pickle
import os

# Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to model file
model_path = os.path.join(script_dir, "fraud_model.pkl")
scaler_path = os.path.join(script_dir, "scaler.pkl")

# Load models
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a transaction CSV file to detect fraudulent entries.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
    data["Amount"] = scaler.transform(data[["Amount"]])
    pred = model.predict(data)
    data["Fraud_Prediction"] = pred
    st.dataframe(data.head())
    st.download_button("Download Results", data.to_csv(index=False), "fraud_results.csv")