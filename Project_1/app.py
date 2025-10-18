import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🛍️ Customer Segmentation for Retail Store")

# Load data: prefer a path relative to this script so the app works when run from other CWDs.
data_path = Path(__file__).resolve().parent / "Mall_Customers.csv"
# also try a common alternate location (project root / Project_1) if launched from workspace root
alt_path = Path.cwd() / "Project_1" / "Mall_Customers.csv"
if not data_path.exists() and alt_path.exists():
    data_path = alt_path

if data_path.exists():
    df = pd.read_csv(data_path)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
else:
    st.error(f"Data file not found. Looking for 'Mall_Customers.csv' in: {data_path}\nYou can upload the CSV below.")
    uploaded = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    else:
        # Stop execution so the rest of the app (which expects `df`) doesn't run and raise further errors.
        st.stop()
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale + cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = reduced[:,0], reduced[:,1]

# Sidebar filters
cluster_choice = st.sidebar.selectbox("Select Cluster", options=sorted(df['Cluster'].unique()))
st.sidebar.write("Number of customers:", (df['Cluster']==cluster_choice).sum())

# Main visualization
col1, col2 = st.columns([3,2])
with col1:
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2', s=80)
    plt.title('Customer Segments')
    st.pyplot(fig)

with col2:
    st.subheader("Cluster Averages")
    cluster_summary = df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean().round(2)
    st.dataframe(cluster_summary)

st.markdown("---")
st.markdown("### 📊 Business Insights")
st.markdown("""
| Cluster | Description | Suggested Action |
|----------|--------------|------------------|
| 0 | Young, High Income, High Spending | Target luxury offers |
| 1 | Middle-aged, Average Income | Loyalty programs |
| 2 | Older, High Income, Low Spending | Retention strategy |
| 3 | Young, Low Income, High Spending | Trendy promotions |
| 4 | Low Income, Low Spending | Discount marketing |
""")