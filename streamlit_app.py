import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


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
    return data, deleted_cols, capped_count


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
    importances = rf.feature_importances_
    feature_names = list(pd.get_dummies(data[features], columns=data[features].select_dtypes(include='object').columns).columns)

    return accuracy, conf_matrix, roc_auc, fpr, tpr, importances, feature_names


def main():
    st.title("Marketing Campaign Analysis with Random Forest")

    file_path = "marketing_campaign1.xlsx"  # Or make this a file uploader
    data = load_data(file_path)
    
    # Feature Engineering with additional info
    data, deleted_cols, capped_count = feature_engineering(data)
    
    # Define features and target
    used_clustering_features = ['Income', 'Age', 'Total_Spending', 'Education', 'Marital_Group',
                             'Children']
    target = 'Response'

    # Train and evaluate the model
    accuracy, conf_matrix, roc_auc, fpr, tpr, importances, feature_names = train_and_evaluate_rf(data, used_clustering_features, target)

    st.header("Model Performance")

    # New UI elements
    st.subheader("Data Insights")
    st.write(f"Number of values capped from IQR in 'Total_Spending': {capped_count}")
    st.write(f"Deleted columns: {', '.join(deleted_cols)}")
    st.write(f"Columns used for model training: {', '.join(used_clustering_features)}")

    st.subheader("Accuracy")
    st.metric("Accuracy", f"{accuracy:.2f}")

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

    # Ensure we have a list of feature names
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis', ax=ax_import)
    ax_import.set_xlabel("Importance Score")
    ax_import.set_ylabel("Features")
    ax_import.set_title("Feature Importance from Random Forest")

    st.pyplot(fig_import)
    
    st.subheader("Modules used")
    st.write("Scikit-learn (RandomForestClassifier, train_test_split, metrics)")
    st.write("Pandas")
    st.write("Numpy")
    st.write("Matplotlib")
    st.write("Seaborn")

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
