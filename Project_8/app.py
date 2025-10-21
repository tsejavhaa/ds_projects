import streamlit as st
import pandas as pd
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model & scaler
model = joblib.load(os.path.join(current_dir, 'logistic_regression_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))

# Try to load saved feature list
feature_names = joblib.load(os.path.join(current_dir, 'feature_list.pkl'))


st.title("📉 Customer Churn Prediction App")

st.markdown("Predict whether a customer is likely to cancel their subscription.")

# --- UI: collect inputs from user ---
st.sidebar.header("Customer inputs")
tenure = st.sidebar.number_input("Customer Tenure (Months)", 0, 200, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
total = st.sidebar.number_input("Total Charges", 0.0, 100000.0, 800.0)
contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
payment = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
dependents = st.sidebar.selectbox("Dependents", ["Yes","No"])

# --- Build an empty input dataframe with all feature columns set to zero ---
# Note: feature_names should match exactly the names used during training (order doesn't strictly matter, but must be same set)
X_input = pd.DataFrame(columns=feature_names)
X_input.loc[0] = 0  # single row, all zeros

# --- Fill numeric features (ensure column names exactly as during training) ---
# Typical numeric columns in the Telco preprocessing example were: 'tenure','MonthlyCharges','TotalCharges'
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    if col in X_input.columns:
        X_input.at[0, col] = [tenure, monthly, total][['tenure','MonthlyCharges','TotalCharges'].index(col)]
    else:
        st.warning(f"Numeric column {col} not found in feature_names; you may need to re-create feature_names.pkl")

# --- Fill categorical one-hot features ---
# Example get_dummies pattern: Contract_Month-to-month, Contract_One year, Contract_Two year
# Map contract to the correct dummy col(s)
contract_map = {
    'Month-to-month': 'Contract_Month-to-month',
    'One year': 'Contract_One year',
    'Two year': 'Contract_Two year'
}
if contract_map[contract] in X_input.columns:
    X_input.at[0, contract_map[contract]] = 1

# PaymentMethod e.g. PaymentMethod_Electronic check
pay_map = {
    'Electronic check': 'PaymentMethod_Electronic check',
    'Mailed check': 'PaymentMethod_Mailed check',
    'Bank transfer (automatic)': 'PaymentMethod_Bank transfer (automatic)',
    'Credit card (automatic)': 'PaymentMethod_Credit card (automatic)'
}
if pay_map[payment] in X_input.columns:
    X_input.at[0, pay_map[payment]] = 1

# Gender example: 'gender_Male' or 'Gender_Male' depending on original columns; try multiple candidates
gender_cols = [c for c in X_input.columns if 'gender' in c.lower() or 'sex' in c.lower()]
if gender_cols:
    # try to set Male/ Female appropriately; check column name suffix/format
    # often get_dummies with drop_first=True creates 'gender_Male' if original column was 'gender'
    for col in gender_cols:
        if 'male' in col.lower():
            X_input.at[0, col] = 1 if gender == "Male" else 0
        elif 'female' in col.lower():
            X_input.at[0, col] = 1 if gender == "Female" else 0

# Dependents example: 'Dependents_Yes'
if f"Dependents_Yes" in X_input.columns:
    X_input.at[0, "Dependents_Yes"] = 1 if dependents == "Yes" else 0

# --- Scale numeric columns with saved scaler (if scaler expects those columns) ---
# scaler was fit on the numeric columns: tenure, MonthlyCharges, TotalCharges
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
present_numeric_cols = [c for c in numeric_cols if c in X_input.columns]

if present_numeric_cols:
    X_input[present_numeric_cols] = scaler.transform(X_input[present_numeric_cols])

# Ensure the dtype is numeric
X_input = X_input.fillna(0).astype(float)

# --- Predict ---
if st.button("Predict churn probability"):
    try:
        pred_prob = model.predict_proba(X_input)[0][1]
        pred_class = "Churn" if pred_prob > 0.5 else "Retain"
        st.subheader(f"Prediction: **{pred_class}**")
        st.metric("Churn probability", f"{pred_prob*100:.2f}%")
    except Exception as e:
        st.error("Prediction failed. This usually means the input features still don't match model features.")
        st.exception(e)