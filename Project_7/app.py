import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("🛒 Walmart Sales Forecasting App")
st.write("Predict weekly sales for a given store and department.")

# Upload data
uploaded = st.file_uploader("Upload sales data (train.csv)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df["Date"] = pd.to_datetime(df["Date"])
    store = st.selectbox("Select Store ID", df["Store"].unique())
    dept = st.selectbox("Select Department ID", df["Dept"].unique())

    data = df[(df["Store"] == store) & (df["Dept"] == dept)][["Date", "Weekly_Sales"]]
    data = data.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)

    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)