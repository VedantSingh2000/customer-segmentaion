import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import plotly.express as px

# -----------------------
# Load and preprocess data
# -----------------------
def load_data():
    df = pd.read_excel("marketing_campaign.xlsx")
    return df


def feature_engineering(df):
    dropped_cols = ['Z_CostContact', 'Z_Revenue', 'ID']
    df.drop(columns=dropped_cols, inplace=True, errors='ignore')
    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    cap_count = (df['Total_Spending'] > 3000).sum()
    df['Total_Spending'] = df['Total_Spending'].apply(lambda x: min(x, 3000))
    out_count = df[df['Income'] > 600000].shape[0]
    df = df[df['Income'] <= 600000]
    return df, dropped_cols, cap_count, out_count


def train_rf(df):
    df = df.dropna()
    df_encoded = pd.get_dummies(df.select_dtypes(include=['object', 'category']), drop_first=True)
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    df_final = pd.concat([df_numeric, df_encoded], axis=1)

    X = df_final.drop('Response', axis=1)
    y = df_final['Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    return accuracy, fpr, tpr, roc_auc, rf.feature_importances_, X.columns


def plot_feature_importance(importances, columns):
    fig, ax = plt.subplots()
    feat_imp = pd.Series(importances, index=columns).sort_values(ascending=False)[:10]
    feat_imp.plot(kind='barh', ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)


def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)


def clustering_graphs(df):
    fig_list = []
    df_cluster = df[['Income', 'Total_Spending']].dropna()

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_cluster['KMeans'] = kmeans.fit_predict(df_cluster)
    fig1 = px.scatter(df_cluster, x='Income', y='Total_Spending', color='KMeans', title='KMeans Clustering')
    fig_list.append(fig1)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=2)
    df_cluster['Agglomerative'] = agg.fit_predict(df_cluster[['Income', 'Total_Spending']])
    fig2 = px.scatter(df_cluster, x='Income', y='Total_Spending', color='Agglomerative', title='Agglomerative Clustering')
    fig_list.append(fig2)

    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    df_cluster['GMM'] = gmm.fit_predict(df_cluster[['Income', 'Total_Spending']])
    fig3 = px.scatter(df_cluster, x='Income', y='Total_Spending', color='GMM', title='GMM Clustering')
    fig_list.append(fig3)

    return fig_list


# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Customer Personality Analysis Dashboard")

    df = load_data()
    df, dropped_cols, cap_count, out_count = feature_engineering(df)

    st.subheader("Data Overview")
    st.write("### Columns dropped:", dropped_cols)
    st.write(f"### Values capped from Total Spending > 3000: {cap_count}")
    st.write(f"### Income outliers removed (Income > 600000): {out_count}")
    st.write("### Final Dataset Shape:", df.shape)

    marital_group = st.multiselect("Filter by Marital Status", options=df['Marital_Status'].unique())
    education_group = st.multiselect("Filter by Education", options=df['Education'].unique())
    income_range = st.slider("Select Income Range", 0, int(df['Income'].max()), (0, int(df['Income'].max())))

    filtered_df = df.copy()
    if marital_group:
        filtered_df = filtered_df[filtered_df['Marital_Status'].isin(marital_group)]
    if education_group:
        filtered_df = filtered_df[filtered_df['Education'].isin(education_group)]
    filtered_df = filtered_df[(filtered_df['Income'] >= income_range[0]) & (filtered_df['Income'] <= income_range[1])]

    st.dataframe(filtered_df.head())

    st.markdown("---")
    st.subheader("Random Forest Model")

    if st.button("Show Accuracy, ROC Curve & Feature Importances"):
        accuracy, fpr, tpr, roc_auc, importances, columns = train_rf(df)
        st.write(f"### Accuracy: {accuracy:.2f}")
        plot_roc_curve(fpr, tpr, roc_auc)
        plot_feature_importance(importances, columns)

    st.markdown("---")
    st.subheader("Clustering Models (K = 2)")

    fig_list = clustering_graphs(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_list[0], use_container_width=True)
    with col2:
        st.plotly_chart(fig_list[1], use_container_width=True)
    with col3:
        st.plotly_chart(fig_list[2], use_container_width=True)


if __name__ == '__main__':
    main()
