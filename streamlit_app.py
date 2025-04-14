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
.graph-container {
    width: 180px;
    height: 140px;
    transition: all 0.3s ease-in-out;
    overflow: hidden;
    margin: 10px;
    border: 1px solid #ddd;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.graph-container:hover {
    width: 400px;
    height: 300px;
    z-index: 10;
}
.title-highlight {
    font-size: 14px;
    font-weight: bold;
    color: #3b82f6;
    text-align: center;
}
.metric-label {
    font-weight: bold;
    color: #16a34a;
}
.sidebar-header {
    font-size: 22px;
    font-weight: bold;
    color: #2563eb;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    return df

def feature_engineering(df):
    df['Age'] = 2025 - df['Year_Birth']
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spending'] = df[spending_cols].sum(axis=1)
    spending_cap = df['Total_Spending'].quantile(0.99)
    capped_count = (df['Total_Spending'] > spending_cap).sum()
    df['Total_Spending'] = np.where(df['Total_Spending'] > spending_cap, spending_cap, df['Total_Spending'])

    median_income = df['Income'].median()
    df['Income'] = df['Income'].fillna(median_income)
    Q1 = df['Income'].quantile(0.25)
    Q3 = df['Income'].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    out_count = ((df['Income'] < low) | (df['Income'] > high)).sum()
    df['Income'] = np.clip(df['Income'], low, high)

    df['Children'] = df['Kidhome'] + df['Teenhome']
    df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)

    # Using Marital_Group as relationship filter
    df['Marital_Group'] = df['Marital_Status'].apply(
        lambda x: 'Single' if x in ['Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd']
        else 'Family'
    )

    df['Education'] = df['Education'].replace({
        'Basic': 'Undergraduate',
        '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate',
        'Master': 'Postgraduate',
        'PhD': 'Postgraduate'
    })

    drop_cols = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Marital_Status']
    df.drop(drop_cols, axis=1, inplace=True)

    return df, drop_cols, capped_count, out_count

def train_rf(data, features, target='Response'):
    # One-hot encode and scale data for modeling
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = classification_report(y_test, y_pred, output_dict=True)['accuracy']
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    return acc, roc_auc, fpr, tpr

def scale_features(data, features_to_scale):
    scaler = StandardScaler()
    scaled_data = data.copy()
    scaled_data[features_to_scale] = scaler.fit_transform(scaled_data[features_to_scale])
    return scaled_data

def plot_scaled_features(scaled_df, features):
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3)) # Adjusted size
    for i, feature in enumerate(features):
        sns.histplot(scaled_df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f"Scaled {feature}")
        axes[i].tick_params(axis='both', which='major', labelsize=8) # Smaller tick labels
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4, 3)) # Adjusted size
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=8) # Smaller labels
    ax.set_ylabel('True Positive Rate', fontsize=8) # Smaller labels
    ax.set_title('Receiver Operating Characteristic', fontsize=10) # Smaller title
    ax.tick_params(axis='both', which='major', labelsize=8) # Smaller tick labels
    ax.legend(fontsize=8) # Smaller legend
    return fig

def clustering_graphs(data):
    # Use the clustering features and run PCA for visualization
    cluster_features = ['Income', 'Age', 'Total_Spending']
    X = StandardScaler().fit_transform(data[cluster_features])
    X_pca = PCA(n_components=2).fit_transform(X)
    data['PCA1'], data['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    figs = {}

    # --- KMeans (k=2) ---
    data['Cluster'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)
    fig, ax = plt.subplots(figsize=(4, 3))  # Adjusted size for better fit
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', ax=ax, s=20) # Smaller points
    ax.set_title("KMeans (k=2)", fontsize=10) # Smaller title
    ax.tick_params(axis='both', which='major', labelsize=8) # Smaller ticks
    ax.set_xlabel("PCA1", fontsize=8) # Smaller labels
    ax.set_ylabel("PCA2", fontsize=8) # Smaller labels
    ax.legend(fontsize=8) # Smaller legend
    figs['KMeans'] = fig

    # --- Agglomerative Clustering (k=2) ---
    data['Cluster'] = AgglomerativeClustering(n_clusters=2).fit_predict(X)
    fig, ax = plt.subplots(figsize=(4, 3))  # Adjusted size for better fit
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='plasma', ax=ax, s=20) # Smaller points
    ax.set_title("Agglomerative (k=2)", fontsize=10) # Smaller title
    ax.tick_params(axis='both', which='major', labelsize=8) # Smaller ticks
    ax.set_xlabel("PCA1", fontsize=8) # Smaller labels
    ax.set_ylabel("PCA2", fontsize=8) # Smaller labels
    ax.legend(fontsize=8) # Smaller legend
    figs['Agglomerative'] = fig

    # --- DBSCAN ---
    data['Cluster'] = DBSCAN(eps=1.2, min_samples=5).fit_predict(X)
    fig, ax = plt.subplots(figsize=(4, 3))  # Adjusted size for better fit
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='cubehelix', ax=ax, s=20) # Smaller points
    ax.set_title("DBSCAN", fontsize=10) # Smaller title
    ax.tick_params(axis='both', which='major', labelsize=8) # Smaller ticks
    ax.set_xlabel("PCA1", fontsize=8) # Smaller labels
    ax.set_ylabel("PCA2", fontsize=8) # Smaller labels
    ax.legend(fontsize=8) # Smaller legend
    figs['DBSCAN'] = fig

    # --- Gaussian Mixture Model (k=2) ---
    data['Cluster'] = GaussianMixture(n_components=2, random_state=42).fit_predict(X)
    fig, ax = plt.subplots(figsize=(4, 3))  # Adjusted size for better fit
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='coolwarm', ax=ax, s=20) # Smaller points
    ax.set_title("GMM (k=2)", fontsize=10) # Smaller title
    ax.tick_params(axis='both', which='major', labelsize=8) # Smaller ticks
    ax.set_xlabel("PCA1", fontsize=8) # Smaller labels
    ax.set_ylabel("PCA2", fontsize=8) # Smaller labels
    ax.legend(fontsize=8) # Smaller legend

    figs['GMM'] = fig

    # Clean up temporary column
    data.drop('Cluster', axis=1, inplace=True)
    return figs

def main():
    st.title("🧠 Customer Segmentation Dashboard")

    # --- Load and transform data ---
    df = load_data("marketing_campaign1.xlsx")
    df, dropped_cols, cap_count, out_count = feature_engineering(df)

    # --- Sidebar Filter Options ---
    st.sidebar.markdown("<div class='sidebar-header'>Filter Options</div>", unsafe_allow_html=True)

    rel_options = list(df["Marital_Group"].unique())
    edu_options = list(df["Education"].unique())

    selected_rel = st.sidebar.multiselect("Select Relationship (Marital Group)", options=rel_options, default=rel_options)
    selected_edu = st.sidebar.multiselect("Select Education Level", options=edu_options, default=edu_options)

    min_income = int(df["Income"].min())
    max_income = int(df["Income"].max())
    selected_income = st.sidebar.slider("Income Range", min_value=min_income, max_value=max_income, value=(min_income, max_income))

    # --- Apply filters to data ---
    filtered_df = df[
        (df["Marital_Group"].isin(selected_rel)) &
        (df["Education"].isin(selected_edu)) &
        (df["Income"] >= selected_income[0]) &
        (df["Income"] <= selected_income[1])
    ]

    # --- Compute main model accuracy and ROC curve using filtered data ---
    used_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    accuracy, roc_auc, fpr, tpr = train_rf(filtered_df, used_features)

    # --- Scale features for visualization ---
    features_to_scale = ['Income', 'Age', 'Total_Spending']
    scaled_df = scale_features(filtered_df.copy(), features_to_scale)
    scaled_features_fig = plot_scaled_features(scaled_df, features_to_scale)
    roc_fig = plot_roc_curve(fpr, tpr, roc_auc)

    # --- Clustering visualizations based on filtered data ---
    cluster_figs = clustering_graphs(filtered_df.copy())

    # --- Display Insights and Performance ---
    st.header("📊 Insights and Model Performance")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📌 Data Overview")
            st.write(f"**Values capped from Total Spending:** {cap_count}")
            st.write(f"**Income outliers removed:** {out_count}")
            st.write(f"**Deleted columns:** {', '.join(dropped_cols)}")
            st.write(f"**Features used for model:** {', '.join(used_features)}")
        with col2:
            st.subheader("✅ Random Forest Performance")
            st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
            st.metric(label="ROC AUC Score", value=f"{roc_auc:.2f}")

    st.divider()

    # --- Display Smaller ROC Curve ---
    st.header("📈 ROC Curve")
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.pyplot(roc_fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- Display Smaller Feature Scaling Plots ---
    st.header("📊 Feature Scaling")
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.pyplot(scaled_features_fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- Display Model Highlights (Clustering Graphs) ---
    st.header("🌀 Clustering Model Highlights")
    st.markdown("Hover over each graph to expand 👇")
    cols = st.columns(4)
    model_names = list(cluster_figs.keys())
    for i, name in enumerate(model_names):
        with cols[i]:
            st.markdown(f"<div class='title-highlight'>{name}</div>", unsafe_allow_html=True)
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.pyplot(cluster_figs[name])
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
