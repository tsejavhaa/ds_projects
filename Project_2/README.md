# 🎬 Movie Recommendation System

This project predicts movie ratings and recommends movies to users using the MovieLens 100K dataset and collaborative filtering (SVD algorithm).

## 🚀 Features
- Exploratory Data Analysis (EDA)
- Collaborative Filtering using SVD
- Movie Recommendation for any user
- Business insights from user behavior
- Streamlit web app for live recommendations

## 📊 Dataset
MovieLens 100K: https://grouplens.org/datasets/movielens/100k/

## 🧠 Model
- Algorithm: Singular Value Decomposition (SVD)
- Library: scikit-learn
- Evaluation: RMSE, MAE (cross-validation)

## 💡 Business Insights
- Drama and Romance genres have the highest average ratings.
- Action and Sci-Fi attract large audiences but receive moderate ratings.
- Opportunity: Focus on quality production in niche genres like Horror or Sci-Fi.

## 🌐 Web App
Run the Streamlit app locally:
```bash
streamlit run app.py