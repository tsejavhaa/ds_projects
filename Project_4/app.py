import streamlit as st
import pandas as pd
import pickle
import os

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would have survived the Titanic disaster.")

# Sidebar for user inputs
st.sidebar.header("Passenger Features")

Pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 3 = 3rd)", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
Age = st.sidebar.slider("Age", 1, 80, 25)
Fare = st.sidebar.slider("Fare", 0, 500, 50)
Embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
FamilySize = st.sidebar.slider("Family Size", 1, 10, 1)
IsAlone = 1 if FamilySize == 1 else 0

# Encode features
sex_map = {"Male": 1, "Female": 0}
embarked_map = {"C": 0, "Q": 1, "S": 2}

# Load trained model and encoders
current_dir = os.path.dirname(os.path.abspath('__file__')) 
model_path = os.path.join(current_dir, 'Project_4/titanic_decision_tree.pkl')

try:
    # Load model and get feature names
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    # Get feature names and create input DataFrame
    feature_names = model_data['features']
    input_data = [[Pclass, sex_map[Sex], Age, Fare, embarked_map[Embarked], FamilySize, IsAlone]]
    X_input = pd.DataFrame(input_data, columns=feature_names)
    
    # Make prediction
    model = model_data['model']
    prediction = model.predict(X_input)[0]
    
    # Display result
    st.subheader("Prediction Result:")
    st.write("✅ Survived" if prediction == 1 else "❌ Did Not Survive")
    
except FileNotFoundError:
    st.error(f"Model file not found at: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or making prediction: {str(e)}")
    st.stop()