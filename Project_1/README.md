# 🛍️ Customer Segmentation for Retail Store

**Author:** Javhaa 
**Tools:** Python, Pandas, Scikit-learn, Seaborn, Streamlit  

---

## 🎯 Project Overview
This project identifies distinct customer groups based on **age**, **annual income**, and **spending behavior** using **unsupervised machine learning (K-Means clustering)**.  
By understanding customer segments, retail businesses can:
- Personalize marketing campaigns  
- Improve customer retention  
- Optimize product recommendations  

---

## 📊 Dataset
**Source:** [Mall Customers Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
**Description:**  
| Column | Description |
|--------|-------------|
| CustomerID | Unique identifier |
| Gender | Male / Female |
| Age | Age of the customer |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Spending behavior score |

---

## 🧠 Objective
Use **K-Means clustering** to find natural groupings among customers and interpret those clusters into actionable business insights.

---

## 🧮 Methods Used
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Scaling (StandardScaler)  
- Clustering with K-Means  
- Model Evaluation (Elbow Method, Silhouette Score)  
- Dimensionality Reduction (PCA) for visualization  
- Business Interpretation of clusters  

---

## 📈 Key Steps

1. **Load & Explore Data**
   - Inspect data types, handle missing values, and encode gender.
2. **Exploratory Analysis**
   - Understand relationships between income, age, and spending score.
3. **Feature Scaling**
   - Normalize numerical features for clustering.
4. **Optimal Cluster Selection**
   - Use the Elbow Method and Silhouette Score.
5. **Cluster Creation**
   - Apply K-Means and assign each customer to a group.
6. **Visualization**
   - 2D plots and PCA visualizations to understand customer groups.
7. **Interpretation**
   - Translate clusters into actionable marketing insights.

---

## 📊 Results & Insights

| Cluster | Characteristics | Business Insight | Marketing Strategy |
|----------|-----------------|------------------|--------------------|
| 0 | Young (25–35), High Income, High Spending | Luxury Shoppers | Premium offers, exclusive launches |
| 1 | Middle-aged, Moderate Income, Average Spending | Balanced Buyers | Loyalty programs, seasonal discounts |
| 2 | Older, High Income, Low Spending | Conservative Spenders | Retention-focused offers |
| 3 | Young, Low Income, High Spending | Trendy Youth | Affordable trendy collections |
| 4 | Low Income, Low Spending | Budget Customers | Essential goods, discounts |

**Silhouette Score:** ~0.43  
**Number of Clusters:** 5  

---

## 📊 Visualizations

| Visualization | Description |
|----------------|--------------|
| 💰 *Income vs Spending* | Displays income and spending relationships across clusters |
| 👥 *PCA Plot* | 2D visualization of clusters |
| 📦 *Cluster Summary Table* | Average metrics for each cluster |

---