import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA  # ✅ PCA import added
import graphviz  # For creating flowcharts
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as sch

# --- Set Page Config MUST be first Streamlit command ---
st.set_page_config(layout="wide")

# --- CSS for Hover Effect ---
st.markdown(
    """
<style>
.small-graph {
    width: 200px;
    height: 150px;
    transition: width 0.3s, height 0.3s;
}
.small-graph:hover {
    width: 400px;
    height: 300px;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    return data

def feature_engineering(data):
    data['Age'] = 2025 - data['Year_Birth']

    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    data['Total_Spending'] = data[spending_cols].sum(axis=1)
    spending_cap = data['Total_Spending'].quantile(0.99)
    capped_count = (data['Total_Spending'] > spending_cap).sum()
    data['Total_Spending'] = np.where(data['Total_Spending'] > spending_cap, spending_cap,
                                      data['Total_Spending'])

    median_income = data['Income'].median()
    data['Income'] = data['Income'].fillna(median_income)

    Q1 = data['Income'].quantile(0.25)
    Q3 = data['Income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    income_outliers_count = len(data[(data['Income'] < lower_bound) | (data['Income'] > upper_bound)])
    data['Income'] = np.clip(data['Income'], lower_bound, upper_bound)

    data['Children'] = data['Kidhome'] + data['Teenhome']
    data.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)

    single_statuses = ['Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd']
    family_statuses = ['Married', 'Together']
    data['Marital_Group'] = data['Marital_Status'].apply(
        lambda x: 'Single' if x in single_statuses else ('Family' if x in family_statuses else x))

    data['Education'] = data['Education'].replace({
        'Basic': 'Undergraduate',
        '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate',
        'Master': 'Postgraduate',
        'PhD': 'Postgraduate'
    })

    deleted_cols = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
    data.drop(deleted_cols, axis=1, inplace=True)

    return data, deleted_cols, capped_count, income_outliers_count

def train_and_evaluate_rf(data, features, target):
    X = data[features]
    y = data[target]
    X = pd.get_dummies(X, columns=X.select_dtypes(include='object').columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                        stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    r_multi = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)
    importances = r_multi.importances_mean
    feature_names = list(pd.get_dummies(data[features], columns=data[features].select_dtypes(include='object').columns).columns)

    return accuracy, conf_matrix, roc_auc, fpr, tpr, importances, feature_names

def run_clustering_and_visualize(data):
    clustering_features = ['Income', 'Age', 'Total_Spending']
    X_orig = data[clustering_features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_orig)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]

    cluster_plots = {}

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['KMeans_Cluster'] = kmeans.fit_predict(X)
    fig_kmeans, ax_kmeans = plt.subplots(figsize=(4, 3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='viridis', ax=ax_kmeans)
    ax_kmeans.set_title('K-Means (k=3)')
    cluster_plots['KMeans'] = fig_kmeans
    data.drop('KMeans_Cluster', axis=1, inplace=True)

    agglo = AgglomerativeClustering(n_clusters=2)
    agglo.fit_predict(X)
    fig_dendro, ax_dendro = plt.subplots(figsize=(4, 3))
    sch.dendrogram(sch.linkage(X, method='ward'), ax=ax_dendro, no_labels=True)
    ax_dendro.set_title('Agglo. Dendrogram (k=2)')
    cluster_plots['Agglomerative'] = fig_dendro

    gmm = GaussianMixture(n_components=2, random_state=42)
    data['GMM_Cluster'] = gmm.fit_predict(X)
    fig_gmm, ax_gmm = plt.subplots(figsize=(4, 3))
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='GMM_Cluster', palette='viridis', ax=ax_gmm)
    ax_gmm.set_title('GMM (k=2)')
    cluster_plots['GMM'] = fig_gmm
    data.drop('GMM_Cluster', axis=1, inplace=True)

    return cluster_plots

def main():
    st.title("Customer Segmentation and Response Prediction")

    file_path = "marketing_campaign1.xlsx"
    data = load_data(file_path)
    data, deleted_cols, capped_count, income_outliers_count = feature_engineering(data)
    used_clustering_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group', 'Children']
    target = 'Response'
    accuracy, conf_matrix, roc_auc, fpr, tpr, importances, feature_names = train_and_evaluate_rf(data, used_clustering_features, target)
    cluster_plots = run_clustering_and_visualize(data)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Insights and Model Performance")
        st.subheader("Data Overview")
        st.write(f"Values capped from Total Spending: **{capped_count}**")
        st.write(f"Income outliers removed: **{income_outliers_count}**")
        st.write(f"Deleted columns: **{', '.join(deleted_cols)}**")
        st.write(f"Features used for model: **{', '.join(used_clustering_features)}**")

        st.subheader("Model Accuracy")
        st.write(f"**Accuracy: {accuracy:.2f}**")

        st.subheader("Machine Learning Models Used")
        st.write("K-Means Clustering, Agglomerative Clustering, DBSCAN, Gaussian Mixture Model, Random Forest Classifier")

    with col2:
        st.header("Model Visualizations")
        st.subheader("Random Forest Classifier")

        fig_size = (6, 4)

        st.subheader("Confusion Matrix")
        fig_conf, ax_conf = plt.subplots(figsize=fig_size)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_conf)
        ax_conf.set_xlabel('Predicted Labels')
        ax_conf.set_ylabel('True Labels')
        ax_conf.set_title('Confusion Matrix')
        st.pyplot(fig_conf)

        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=fig_size)
        ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        st.pyplot(fig_roc)

        st.subheader("Feature Importance")
        fig_import, ax_import = plt.subplots(figsize=fig_size)
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis', ax=ax_import)
        ax_import.set_xlabel("Importance Score")
        ax_import.set_ylabel("Features")
        ax_import.set_title("Feature Importance")
        st.pyplot(fig_import)

        st.subheader("Clustering Visualizations")
        col_layout = st.columns(len(cluster_plots))
        for i, (name, fig) in enumerate(cluster_plots.items()):
            with col_layout[i]:
                st.pyplot(fig, use_container_width=False, output_format='auto', figure_class='small-graph')

    st.subheader("Module Flowchart")
    graph = graphviz.Digraph(comment='Module Flowchart')
    graph.node('A', 'Load Data')
    graph.node('B', 'Feature Engineering')
    graph.node('C', 'Train/Evaluate RF')
    graph.edge('A', 'B', label='Data')
    graph.edge('B', 'C', label='Engineered Features')
    st.graphviz_chart(graph)

if __name__ == "__main__":
    main()
