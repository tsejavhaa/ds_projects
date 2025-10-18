# 🛍️ Sentiment Analysis on Amazon Product Reviews (Memory-Safe)

### 📘 Overview
This project predicts whether an Amazon product review is positive or negative.  
Optimized for limited-memory systems (like M1 Macs) by streaming only a subset of the dataset.

### ⚙️ Tech Stack
- Python (pandas, numpy, scikit-learn)
- NLTK for text preprocessing
- Logistic Regression & Naive Bayes
- Streamlit for interactive web app

### 📂 Dataset
[Amazon Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews)  
Uses `train.ft.txt` and `test.ft.txt` files — loaded efficiently via custom function.

### 🚀 Run Locally
```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib streamlit
python -m nltk.downloader stopwords
streamlit run sentiment_app.py