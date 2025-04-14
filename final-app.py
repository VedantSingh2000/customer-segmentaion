import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# --- Page config ---
st.set_page_config(layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    .stPlot {
        width: 100% !important;
    }
    .cluster-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Dummy function: Load data ---
def load_data(path):
    return pd.read_excel(path)

# --- Feature engineering function ---
def feature_engineering(df):
    cap_count = (df['Total_Spending'] > 3000).sum()
    df['Total_Spending'] = np.where(df['Total_Spending'] > 3000, 3000, df['Total_Spending'])

    q1 = df['Income'].quantile(0.25)
    q3 = df['Income'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['Income'] < (q1 - 1.5 * iqr)) | (df['Income'] > (q3 + 1.5 * iqr))]
    out_count = outliers.shape[0]
    df = df.drop(outliers.index)

    dropped_cols = ['ID', 'Z_CostContact', 'Z_Revenue']
    for col in dropped_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df['Age'] = 2024 - df['Year_Birth']
    df.drop(columns=['Year_Birth'], inplace=True)

    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

    df['Marital_Group'] = df['Marital_Status'].replace({
        'Married': 'Partnered', 'Together': 'Partnered',
        'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single'
    })

    return df, dropped_cols, cap_count, out_count

# --- Feature scaling ---
def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

# --- Train Random Forest ---
def train_rf(df, features):
    df = df.dropna()
    df_encoded = pd.get_dummies(df[features + ['Response']], drop_first=True)
    X = df_encoded.drop(columns='Response')
    y = df_encoded['Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    importance = list(zip(X.columns, model.feature_importances_))
    return accuracy, roc_auc, fpr, tpr, importance

# --- ROC Plot ---
def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4, 3))
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

# --- Feature Importance Plot ---
def plot_feature_importance(importances):
    fig, ax = plt.subplots(figsize=(4, 3))
    importances.sort(key=lambda x: x[1], reverse=True)
    labels, scores = zip(*importances)
    ax.barh(labels, scores, color='skyblue')
    ax.set_title("Feature Importance", fontsize=10)
    ax.set_xlabel("Importance", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig

# --- Clustering Plots ---
def clustering_graphs(data):
    cluster_features = ['Income', 'Age', 'Total_Spending']
    X = StandardScaler().fit_transform(data[cluster_features])
    X_pca = PCA(n_components=2).fit_transform(X)
    data['PCA1'], data['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    figs = {}
    cluster_figsize = (4, 3)

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

    # GMM
    data['Cluster'] = GaussianMixture(n_components=2, random_state=42).fit_predict(X)
    fig, ax = plt.subplots(figsize=cluster_figsize)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='cool', ax=ax, s=30)
    ax.set_title("GMM (k=2)", fontsize=10)
    figs['GMM'] = fig

    data.drop('Cluster', axis=1, inplace=True)
    return figs

# --- Main ---
def main():
    st.title("🧠 Customer Segmentation Dashboard")

    df = load_data("marketing_campaign1.xlsx")
    df, dropped_cols, cap_count, out_count = feature_engineering(df)

    # --- Sidebar Filters ---
    st.sidebar.markdown("### Filter Options")
    rel_options = list(df["Marital_Group"].unique())
    edu_options = list(df["Education"].unique())
    selected_rel = st.sidebar.multiselect("Select Relationship (Marital Group)", options=rel_options, default=rel_options)
    selected_edu = st.sidebar.multiselect("Select Education Level", options=edu_options, default=edu_options)
    min_income = int(df["Income"].min())
    max_income = int(df["Income"].max())
    selected_income = st.sidebar.slider("Income Range", min_value=min_income, max_value=max_income, value=(min_income, max_income))

    # Apply filters
    filtered_df = df[
        (df["Marital_Group"].isin(selected_rel)) &
        (df["Education"].isin(selected_edu)) &
        (df["Income"] >= selected_income[0]) &
        (df["Income"] <= selected_income[1])
    ]

    # Model training
    used_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    accuracy, roc_auc, fpr, tpr, feature_importances = train_rf(filtered_df, used_features)

    # Create visuals
    scaled_df = scale_features(filtered_df.copy(), ['Income', 'Age', 'Total_Spending'])
    cluster_figs = clustering_graphs(filtered_df.copy())
    roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
    feature_fig = plot_feature_importance(feature_importances)

    # --- Insights section ---
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

    st.header("📈 Model Visuals")

    if st.button("📈 Show ROC Curve"):
        st.pyplot(roc_fig, use_container_width=True)

    if st.button("📊 Show Feature Importance"):
        st.pyplot(feature_fig, use_container_width=True)

    st.divider()

    st.header("🌀 Clustering Results")
    cols = st.columns(3)
    for i, (name, fig) in enumerate(cluster_figs.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
