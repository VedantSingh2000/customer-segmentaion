
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Clustering App", layout="centered")
st.title("Customer Personality Clustering")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_customer_data.csv")  # Replace with your actual cleaned data file

data = load_data()

# Feature selection
features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
            'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'Total_Spending']

X = data[features]

# Sidebar
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox("Choose a Clustering Model", ["KMeans"])  # Can add more later
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)

# Apply clustering
if model_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(X)
    data["Cluster"] = cluster_labels
    score = silhouette_score(X, cluster_labels)

    st.subheader("Results")
    st.write(f"Model Used: {model_choice}")
    st.write(f"Number of Clusters: {n_clusters}")
    st.write(f"Silhouette Score (as Accuracy Metric): {score:.4f}")

    # Display Cluster Count
    st.write("Cluster Distribution:")
    st.bar_chart(data["Cluster"].value_counts())

    # Show sample output
    st.subheader("Sample Clustered Data")
    st.dataframe(data.head())

    # Optional: Cluster Visualization (using 2 features)
    st.subheader("Cluster Visualization (2D)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[features[0]], y=data[features[1]], hue=data["Cluster"], palette="Set2", ax=ax)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    st.pyplot(fig)
