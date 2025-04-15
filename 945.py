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

st.cache_data.clear()
st.cache_resource.clear()


# --- Set page configuration ---
st.set_page_config(layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    .cluster-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Data loading and feature engineering ---
def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

def feature_engineering(df):
    dropped_cols = []

    # Create Total_Spending first
    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

    # Cap Total_Spending
    cap_count = (df['Total_Spending'] > 3000).sum()
    df['Total_Spending'] = np.where(df['Total_Spending'] > 3000, 3000, df['Total_Spending'])

    # Remove income outliers
    out_count = ((df['Income'] > 600000)).sum()
    df = df[df['Income'] <= 600000]

    # Age
    df['Age'] = 2024 - df['Year_Birth']
    dropped_cols.append('Year_Birth')
    
    # Marital Group
    df['Marital_Group'] = df['Marital_Status'].replace({
        'Married': 'Partnered', 'Together': 'Partnered',
        'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'
    })
    dropped_cols.append('Marital_Status')

    df.drop(columns=dropped_cols, inplace=True)
    return df, dropped_cols, cap_count, out_count

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def train_rf(df, features):
    df = df.dropna()
    X = pd.get_dummies(df[features], drop_first=True)
    y = (df['Response'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    accuracy = clf.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return accuracy, roc_auc, fpr, tpr, X.columns

def plot_scaled_features(scaled_df, features):
    num_features = len(features)
    fig, axes = plt.subplots(1, num_features, figsize=(8, 3))
    if num_features == 1:
        axes = [axes]
    for i, feature in enumerate(features):
        sns.histplot(scaled_df[feature], kde=True, ax=axes[i], line_kws={'linewidth': 1})
        axes[i].set_title(f"Scaled {feature}", fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    return fig

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
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='coolwarm', ax=ax, s=30)
    ax.set_title("Agglomerative (k=2)", fontsize=10)
    figs['Agglomerative'] = fig

    # GMM
    data['Cluster'] = GaussianMixture(n_components=2, random_state=42).fit_predict(X)
    fig, ax = plt.subplots(figsize=cluster_figsize)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax, s=30)
    ax.set_title("GMM (k=2)", fontsize=10)
    figs['GMM'] = fig

    data.drop('Cluster', axis=1, inplace=True)
    return figs

def main():
    st.title("🧠 Customer Segmentation Dashboard")

    df = load_data("marketing_campaign1.xlsx")
    df, dropped_cols, cap_count, out_count = feature_engineering(df)

    st.sidebar.markdown("### Filter Options")
    rel_options = list(df["Marital_Group"].unique())
    edu_options = list(df["Education"].unique())
    selected_rel = st.sidebar.multiselect("Select Relationship (Marital Group)", options=rel_options, default=rel_options)
    selected_edu = st.sidebar.multiselect("Select Education Level", options=edu_options, default=edu_options)
    min_income = int(df["Income"].min())
    max_income = int(df["Income"].max())
    selected_income = st.sidebar.slider("Income Range", min_value=min_income, max_value=max_income, value=(min_income, max_income))

    filtered_df = df[
        (df["Marital_Group"].isin(selected_rel)) &
        (df["Education"].isin(selected_edu)) &
        (df["Income"] >= selected_income[0]) &
        (df["Income"] <= selected_income[1])
    ]

    used_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    accuracy, roc_auc, fpr, tpr, n_features = train_rf(filtered_df, used_features)

    features_to_scale = ['Income', 'Age', 'Total_Spending']
    scaled_df = scale_features(filtered_df.copy(), features_to_scale)
    scaled_features_fig = plot_scaled_features(scaled_df, features_to_scale)
    roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
    cluster_figs = clustering_graphs(filtered_df.copy())

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
    st.header("📈 Additional Graphs")

    if st.button("Show ROC Curve"):
        st.pyplot(roc_fig)

    if st.button("Show Feature Scaling Graphs"):
        st.pyplot(scaled_features_fig)

    st.divider()
    st.header("🌀 Clustering Results (k=2)")
    cols = st.columns(3)
    for i, name in enumerate(['KMeans', 'Agglomerative', 'GMM']):
        with cols[i]:
            st.markdown(f"<div class='cluster-container'>", unsafe_allow_html=True)
            st.markdown(f"**{name}**")
            st.pyplot(cluster_figs[name])
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
