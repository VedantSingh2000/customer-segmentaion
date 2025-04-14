import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance  # For feature importance


@st.cache_data
def load_data(file_path):
    """Loads data from Excel file and performs initial data cleaning."""
    data = pd.read_excel(file_path)
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    return data


def feature_engineering(data):
    """Performs feature engineering on the data."""
    data['Age'] = 2025 - data['Year_Birth']

    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    data['Total_Spending'] = data[spending_cols].sum(axis=1)
    spending_cap = data['Total_Spending'].quantile(0.99)

    # Add count of values capped in 'Total_Spending'
    capped_count = (data['Total_Spending'] > spending_cap).sum()

    data['Total_Spending'] = np.where(data['Total_Spending'] > spending_cap, spending_cap,
                                        data['Total_Spending'])
    # Impute missing Income values with median
    median_income = data['Income'].median()
    data['Income'] = data['Income'].fillna(median_income)
    # Outlier handling for Income (IQR method)
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

def create_eda_plots(data):
    """Generates EDA plots for the data."""
    eda_plots = {}

    # 1. Distribution of Total Spending
    fig_spending, ax_spending = plt.subplots()
    sns.histplot(data['Total_Spending'], kde=True, ax=ax_spending)
    ax_spending.set_title('Distribution of Total Spending')
    eda_plots['Total Spending Distribution'] = fig_spending

    # 2. Distribution of Age
    fig_age, ax_age = plt.subplots()
    sns.histplot(data['Age'], kde=True, ax=ax_age)
    ax_age.set_title('Distribution of Age')
    eda_plots['Age Distribution'] = fig_age

    # 3. Count plot of Marital Groups
    fig_marital, ax_marital = plt.subplots()
    sns.countplot(x='Marital_Group', data=data, ax=ax_marital)
    ax_marital.set_title('Count of Marital Groups')
    eda_plots['Marital Group Counts'] = fig_marital

    # 4. Boxplot of Income by Education Level
    fig_income_edu, ax_income_edu = plt.subplots()
    sns.boxplot(x='Education', y='Income', data=data, ax=ax_income_edu)
    ax_income_edu.set_title('Income by Education Level')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    eda_plots['Income by Education'] = fig_income_edu

    return eda_plots

def train_and_evaluate_rf(data, features, target):
    """Trains and evaluates a Random Forest model."""
    X = data[features]
    y = data[target]

    # One-Hot Encode categorical columns
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

    # Permutation Importance
    r_multi = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)
    importances = r_multi.importances_mean
    feature_names = list(pd.get_dummies(data[features], columns=data[features].select_dtypes(include='object').columns).columns)


    return accuracy, conf_matrix, roc_auc, fpr, tpr, importances, feature_names


def main():
    st.set_page_config(layout="wide")  # Use the full page width

    st.title("Marketing Campaign Analysis with Random Forest")

    file_path = "marketing_campaign1.xlsx"  # Or make this a file uploader
    data = load_data(file_path)

    # Feature Engineering with additional info
    data, deleted_cols, capped_count, income_outliers_count = feature_engineering(data)

    # Define features and target
    used_clustering_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group',
                             'Children']
    target = 'Response'

    # Train and evaluate the model
    accuracy, conf_matrix, roc_auc, fpr, tpr, importances, feature_names = train_and_evaluate_rf(data, used_clustering_features, target)

    # --- Layout with Columns ---
    col1, col2 = st.columns(2)  # Two main columns

    # --- Column 1: Data Insights and Performance Metrics ---
    with col1:
        st.header("Data Insights & Model Performance")

        st.subheader("Data Overview")
        st.write(f"Number of values capped from Total Spending: **{capped_count}**")
        st.write(f"Number of income outliers removed: **{income_outliers_count}**")
        st.write(f"Deleted columns: **{', '.join(deleted_cols)}**")
        st.write(f"Columns used for model training: **{', '.join(used_clustering_features)}**")

        st.subheader("Model Accuracy")
        st.metric(label="Accuracy", value=f"{accuracy:.2f}")

        st.subheader("Modules Used")
        st.write("Scikit-learn (RandomForestClassifier, train_test_split, metrics)")
        st.write("Pandas")
        st.write("Numpy")
        st.write("Matplotlib")
        st.write("Seaborn")

    # --- Column 2: Visualizations ---
    with col2:
        st.header("Model Visualizations")

        st.subheader("Confusion Matrix")
        fig_conf, ax_conf = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_conf)
        ax_conf.set_xlabel('Predicted Labels')
        ax_conf.set_ylabel('True Labels')
        ax_conf.set_title('Confusion Matrix')
        st.pyplot(fig_conf)

        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        st.pyplot(fig_roc)

        st.subheader("Feature Importance")
        fig_import, ax_import = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
        indices = np.argsort(importances)[::-1]

        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices],
                    palette='viridis', ax=ax_import)
        ax_import.set_xlabel("Importance Score")
        ax_import.set_ylabel("Features")
        ax_import.set_title("Feature Importance from Random Forest")

        st.pyplot(fig_import)
    with st.expander("Explore other graphs"):
        eda_plots = create_eda_plots(data)
        for title, fig in eda_plots.items():
            st.subheader(title)
            st.pyplot(fig)

    # CSS for hover effect (dynamic tile color)
    st.markdown(
        """
        <style>
        .stMetric {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px;
        }

        .stMetric:hover {
            background-color: #ff0000; /* Red on hover */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
