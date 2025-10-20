# 🛒 Walmart Sales Forecasting

## 🎯 Objective
Forecast future store sales to optimize inventory management and reduce stockouts.

## 📊 Dataset
[Kaggle: Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

## ⚙️ Project Steps
1. Data merging and preprocessing  
2. Seasonal decomposition (trend, seasonality, residuals)  
3. Train SARIMA and Prophet models  
4. Evaluate using MAE and RMSE  
5. Visualize and interpret results  
6. Deploy Streamlit dashboard  

## 📈 Results
| Model | MAE | RMSE |
|--------|-----|------|
| SARIMA | ~1100 | ~2000 |
| Prophet | ~1200 | ~1500 |

## 💡 Business Insights
- Yearly and holiday seasonality affect sales.  
- Prophet captures spikes and dips well.  
- Forecasting helps in maintaining optimal inventory.  

## 🧩 Technologies
Python, Pandas, Prophet, Statsmodels, Matplotlib, Streamlit