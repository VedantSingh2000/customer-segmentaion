import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# --- Set page configuration ---
st.set_page_config(layout="wide")

# --- Custom CSS for hover effects and styling ---
st.markdown("""
<style>
    /* Main container styling */
    .stPlotContainer {
        position: relative;
    }
    
    /* Upper graphs container (ROC and Feature Scaling) */
    .upper-graph-container {
        width: 100%;
        height: 300px;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    /* Lower graphs container (Clustering) */
    .lower-graph-container {
        width: 100%;
        height: 250px;
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    
    /* Hover effects */
    .graph-container:hover {
        transform: scale(1.05);
        z-index: 100;
    }
    
    /* Graph titles */
    .graph-title {
        font-size: 14px;
        font-weight: bold;
        color: #3b82f6;
        text-align: center;
        margin: 5px 0;
    }
    
    /* Make sure plots fit their containers */
    .stPlot {
        width: 100% !important;
        height: 100% !important;
    }
    
    /* Adjust figure sizes in matplotlib */
    .small-fig {
        width: 100%;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# [Rest of your existing functions remain exactly the same until the plotting functions]

def plot_scaled_features(scaled_df, features):
    num_features = len(features)
    fig, axes = plt.subplots(1, num_features, figsize=(8, 3))  # Adjusted size for upper section
    if num_features == 1:
        axes = [axes]
    for i, feature in enumerate(features):
        sns.histplot(scaled_df[feature], kde=True, ax=axes[i], line_kws={'linewidth': 1})
        axes[i].set_title(f"Scaled {feature}", fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4, 3))  # Smaller size for side-by-side display
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title('ROC Curve', fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig

def clustering_graphs(data):
    cluster_features = ['Income', 'Age', 'Total_Spending']
    X = StandardScaler().fit_transform(data[cluster_features])
    X_pca = PCA(n_components=2).fit_transform(X)
    data['PCA1'], data['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    figs = {}

    # All clustering plots use this size
    cluster_figsize = (4, 3)  # Smaller size for lower section
    
    # KMeans
    data['Cluster'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)
    fig, ax = plt.subplots(figsize=cluster_figsize)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', ax=ax, s=30)
    ax.set_title("KMeans (k=2)", fontsize=10)
    figs['KMeans'] = fig

    # Agglomerative
    data['Cluster'] = AgglomerativeClustering(n_clusters=2).fit_predict(X)
    fig, ax = plt.subplots(figsize=cluster_figsize)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='plasma', ax=ax, s=30)
    ax.set_title("Agglomerative (k=2)", fontsize=10)
    figs['Agglomerative'] = fig

    # DBSCAN
    data['Cluster'] = DBSCAN(eps=1.2, min_samples=5).fit_predict(X)
    fig, ax = plt.subplots(figsize=cluster_figsize)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='cubehelix', ax=ax, s=30)
    ax.set_title("DBSCAN", fontsize=10)
    figs['DBSCAN'] = fig

    # GMM
    data['Cluster'] = GaussianMixture(n_components=2, random_state=42).fit_predict(X)
    fig, ax = plt.subplots(figsize=cluster_figsize)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='coolwarm', ax=ax, s=30)
    ax.set_title("GMM (k=2)", fontsize=10)
    figs['GMM'] = fig

    data.drop('Cluster', axis=1, inplace=True)
    return figs

def main():
    st.title("🧠 Customer Segmentation Dashboard")

    # --- Load and process data ---
    df = load_data("marketing_campaign1.xlsx")
    df, dropped_cols, cap_count, out_count = feature_engineering(df)

    # --- Sidebar filters ---
    st.sidebar.markdown("### Filter Options")
    rel_options = list(df["Marital_Group"].unique())
    edu_options = list(df["Education"].unique())
    selected_rel = st.sidebar.multiselect("Select Relationship (Marital Group)", options=rel_options, default=rel_options)
    selected_edu = st.sidebar.multiselect("Select Education Level", options=edu_options, default=edu_options)
    min_income = int(df["Income"].min())
    max_income = int(df["Income"].max())
    selected_income = st.sidebar.slider("Income Range", min_value=min_income, max_value=max_income, value=(min_income, max_income))

    # --- Apply filters ---
    filtered_df = df[
        (df["Marital_Group"].isin(selected_rel)) &
        (df["Education"].isin(selected_edu)) &
        (df["Income"] >= selected_income[0]) &
        (df["Income"] <= selected_income[1])
    ]

    # --- Model training ---
    used_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    accuracy, roc_auc, fpr, tpr, n_features = train_rf(filtered_df, used_features)

    # --- Create plots ---
    features_to_scale = ['Income', 'Age', 'Total_Spending']
    scaled_df = scale_features(filtered_df.copy(), features_to_scale)
    scaled_features_fig = plot_scaled_features(scaled_df, features_to_scale)
    roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
    cluster_figs = clustering_graphs(filtered_df.copy())

    # --- Main display ---
    st.header("📊 Insights and Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Data Overview")
        st.write(f"**Values capped from Total Spending:** {cap_count}")
        st.write(f"**Income outliers removed:** {out_count}")
        st.write(f"**Features used:** {', '.join(used_features)}")
    with col2:
        st.subheader("✅ Random Forest Performance")
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        st.metric("ROC AUC Score", f"{roc_auc:.2f}")

    st.divider()

    # --- Upper graphs (ROC and Feature Scaling) ---
    st.header("📉 Model Evaluation & Feature Scaling")
    col_roc, col_scaling = st.columns(2)
    
    with col_roc:
        st.markdown("<div class='graph-title'>📈 ROC Curve</div>", unsafe_allow_html=True)
        st.pyplot(roc_fig)
    
    with col_scaling:
        st.markdown("<div class='graph-title'>📊 Feature Scaling</div>", unsafe_allow_html=True)
        st.pyplot(scaled_features_fig)

    st.divider()

    # --- Lower graphs (Clustering) ---
    st.header("🌀 Clustering Model Highlights")
    st.markdown("Hover over each graph to expand 👇")
    
    cols = st.columns(4)
    for i, (name, fig) in enumerate(cluster_figs.items()):
        with cols[i]:
            st.markdown(f"<div class='graph-title'>{name}</div>", unsafe_allow_html=True)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
