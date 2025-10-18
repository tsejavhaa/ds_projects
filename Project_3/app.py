import streamlit as st
import pandas as pd
import os
import pickle
from xgboost import XGBRegressor

# Constants
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "xgb_house_price.model")
FEATURES_PATH = os.path.join(current_dir, "model_features.pkl")
DATASET_PATH = os.path.join(current_dir, "dataset", "train.csv")

def preprocess_data(df):
    """Preprocess the data consistently for both training and prediction"""
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Convert categorical variables to dummy variables
    df_encoded = pd.get_dummies(df, drop_first=True)  # Use drop_first to avoid multicollinearity
    
    return df_encoded

@st.cache_resource
def load_or_train_model():
    """Load the pre-trained model or train a new one if it doesn't exist"""
    model_info = {'model': None, 'features': None, 'error': None}
    
    try:
        # Try to load existing model and features
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            model = XGBRegressor()
            model.load_model(MODEL_PATH)
            with open(FEATURES_PATH, 'rb') as f:
                model_info['features'] = pickle.load(f)
            model_info['model'] = model
            st.sidebar.success("✅ Loaded pre-trained model")
            return model_info
        
        st.sidebar.info("⏳ Training new model...")
        
        # Load and preprocess data
        df = pd.read_csv(DATASET_PATH)
        df_encoded = preprocess_data(df)
        
        # Get feature columns (excluding target)
        feature_columns = df_encoded.drop('SalePrice', axis=1).columns.tolist()
        
        # Prepare features and target
        X = df_encoded[feature_columns]
        y = df_encoded['SalePrice']
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Save model and features
        model.save_model(MODEL_PATH)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        model_info['model'] = model
        model_info['features'] = feature_columns
        st.sidebar.success("✅ Trained and saved new model")
        
        return model_info
        
    except Exception as e:
        model_info['error'] = str(e)
        return model_info

st.title("🏠 House Price Prediction App")
st.write("Enter details to predict the house price based on trained XGBoost model.")

# Load or train model
model_info = load_or_train_model()

if model_info['error']:
    st.error(f"Error: {model_info['error']}")
else:
    try:
        # Load sample data for reference
        df = pd.read_csv(DATASET_PATH)
        
        # Create tabs for input groups
        tab1, tab2, tab3 = st.tabs(["Basic Info", "Quality Metrics", "Additional Features"])
        
        with tab1:
            st.subheader("Basic Information")
            MSSubClass = st.selectbox("Building Class", sorted(df['MSSubClass'].unique()), 
                                    format_func=lambda x: f"Class {x}")
            MSZoning = st.selectbox("Zoning Classification", sorted(df['MSZoning'].unique()))
            LotArea = st.number_input("Lot Area (sq ft)", 1000, 200000, 8000)
            YearBuilt = st.number_input("Year Built", 1800, 2024, 2000)
            
        with tab2:
            st.subheader("Quality and Condition")
            OverallQual = st.slider("Overall Quality", 1, 10, 5, 
                                help="1: Very Poor, 10: Very Excellent")
            OverallCond = st.slider("Overall Condition", 1, 10, 5,
                                help="1: Very Poor, 10: Very Excellent")
            
        with tab3:
            st.subheader("Additional Features")
            GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
            TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
            GarageCars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
        
        if st.button("Predict Price"):
            # Create input DataFrame with user inputs
            input_data = pd.DataFrame({
                'MSSubClass': [MSSubClass],
                'MSZoning': [MSZoning],
                'LotArea': [LotArea],
                'YearBuilt': [YearBuilt],
                'OverallQual': [OverallQual],
                'OverallCond': [OverallCond],
                'GrLivArea': [GrLivArea],
                'TotalBsmtSF': [TotalBsmtSF],
                'GarageCars': [GarageCars]
            })
            
            # Preprocess input data
            input_encoded = preprocess_data(input_data)
            
            # Handle missing columns efficiently
            missing_cols = set(model_info['features']) - set(input_encoded.columns)
            if missing_cols:
                # Create a DataFrame with zeros for missing columns
                missing_df = pd.DataFrame(0, index=input_encoded.index, columns=list(missing_cols))
                # Concatenate with existing DataFrame
                input_encoded = pd.concat([input_encoded, missing_df], axis=1)
            
            # Reorder columns to match training data
            input_encoded = input_encoded[model_info['features']]
            
            # Make prediction
            prediction = model_info['model'].predict(input_encoded)[0]
            
            # Display result
            st.success(f"Predicted House Price: ${prediction:,.2f}")
            
            # Show feature importance plot
            if st.checkbox("Show Feature Importance"):
                feature_importance = pd.DataFrame({
                    'feature': model_info['features'],
                    'importance': model_info['model'].feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
                
                st.bar_chart(feature_importance.set_index('feature'))
            
            # Make prediction
            prediction = model_info['model'].predict(input_encoded)
            st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
            
            # Show feature importance
            if hasattr(model_info['model'], 'feature_importances_'):
                st.sidebar.markdown("### Top Important Features")
                feature_importance = pd.DataFrame({
                    'Feature': model_info['features'],
                    'Importance': model_info['model'].feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.sidebar.dataframe(feature_importance, hide_index=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

    st.sidebar.markdown("""
    ### Model Information
    This model uses multiple features to predict house prices, including:
    - Building class and zoning
    - Lot and living area
    - Quality and condition ratings
    - Year built
    - Garage and basement features
    
    The prediction is based on historical house sales data.
    """)