import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io  # For capturing df.info() output
import scipy.cluster.hierarchy as sch

# Clustering & dimensionality reduction libraries
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Supervised learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# --- Streamlit Configuration ---
st.set_page_config(layout="wide") # Use wide layout
warnings.filterwarnings("ignore") # Ignore warnings
sns.set(style="whitegrid")
# Suppress pyplot global use warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Helper Functions with Caching ---

@st.cache_data # Cache the data loading and initial processing
def load_data(uploaded_file):
    """Loads data from the uploaded Excel file."""
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.success("File uploaded and loaded successfully!")
            return data
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        # Fallback to default file if needed (for local testing)
        try:
            data = pd.read_excel("marketing_campaign1.xlsx")
            st.info("No file uploaded. Using default 'marketing_campaign1.xlsx'.")
            return data
        except FileNotFoundError:
            st.error("Default file 'marketing_campaign1.xlsx' not found. Please upload a file.")
            return None
        except Exception as e:
            st.error(f"Error loading default file: {e}")
            return None

@st.cache_data # Cache the data cleaning and feature engineering steps
def preprocess_data(data):
    """Cleans data, handles missing values/outliers, and engineers features."""
    df = data.copy() # Work on a copy

    # Convert date column
    try:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
        newest_date = df['Dt_Customer'].max()
        oldest_date = df['Dt_Customer'].min()
    except Exception as e:
        st.warning(f"Could not parse 'Dt_Customer'. Error: {e}. Skipping date processing.")
        newest_date = "N/A"
        oldest_date = "N/A"


    # Impute missing Income
    if 'Income' in df.columns and df['Income'].isnull().any():
        median_income = df['Income'].median()
        df['Income'] = df['Income'].fillna(median_income)
        st.write(f"Filled missing Income values with median: {median_income:.2f}")
    else:
        st.write("No missing 'Income' values found or 'Income' column missing.")

    # Handle outliers in Income
    if 'Income' in df.columns:
        Q1 = df['Income'].quantile(0.25)
        Q3 = df['Income'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        st.write(f"Income outlier handling: Lower bound={lower_bound:.2f}, Upper bound={upper_bound:.2f}")
        df['Income'] = np.clip(df['Income'], lower_bound, upper_bound)
    else:
         st.write("'Income' column not found for outlier handling.")

    # Feature Engineering
    if 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth'] # Using a fixed future year for consistency
    else:
        st.warning("'Year_Birth' column not found. Cannot calculate 'Age'.")
        df['Age'] = np.nan # Add Age column as NaN if Year_Birth is missing

    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    # Check if all spending columns exist
    existing_spending_cols = [col for col in spending_cols if col in df.columns]
    if len(existing_spending_cols) > 0:
        df['Total_Spending'] = df[existing_spending_cols].sum(axis=1)
        # Cap total spending at 99th percentile
        spending_cap = df['Total_Spending'].quantile(0.99)
        df['Total_Spending'] = np.where(df['Total_Spending'] > spending_cap, spending_cap, df['Total_Spending'])
        st.write("Created 'Total_Spending' feature.")
    else:
        st.warning("Could not find sufficient spending columns to calculate 'Total_Spending'.")
        df['Total_Spending'] = 0 # Assign 0 if columns are missing

    # Combine Kidhome and Teenhome
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
        df['Children'] = df['Kidhome'] + df['Teenhome']
        df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True, errors='ignore')
        st.write("Combined 'Kidhome' and 'Teenhome' into 'Children'.")
    else:
        st.warning("Could not find 'Kidhome' or 'Teenhome'. Skipping 'Children' creation.")
        df['Children'] = 0 # Assign 0 if columns are missing

    # Group Marital Status
    if 'Marital_Status' in df.columns:
        single_statuses = ['Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd']
        family_statuses = ['Married', 'Together']
        df['Marital_Group'] = df['Marital_Status'].apply(lambda x: 'Single' if x in single_statuses
                                                          else ('Family' if x in family_statuses else 'Other')) # Added 'Other' for robustness
        st.write("Grouped 'Marital_Status' into 'Marital_Group'.")
    else:
         st.warning("Could not find 'Marital_Status'. Skipping 'Marital_Group' creation.")

    # Education segmentation
    if 'Education' in df.columns:
        df['Education'] = df['Education'].replace({
            'Basic': 'Undergraduate',
            '2n Cycle': 'Undergraduate',
            'Graduation': 'Graduate',
            'Master': 'Postgraduate',
            'PhD': 'Postgraduate'
        })
        st.write("Segmented 'Education' levels.")
    else:
        st.warning("Could not find 'Education'. Skipping education segmentation.")

    # Drop redundant columns (handle potential missing columns gracefully)
    cols_to_drop = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Marital_Status']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df.drop(existing_cols_to_drop, axis=1, inplace=True)
        st.write(f"Dropped redundant columns: {', '.join(existing_cols_to_drop)}")

    return df, newest_date, oldest_date, existing_spending_cols

# --- Main App Logic ---
st.title("Marketing Campaign Analysis and Customer Segmentation")

# --- Sidebar ---
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your Marketing Campaign Excel file", type=["xlsx"])

# --- Data Loading and Preprocessing ---
raw_data = load_data(uploaded_file)

if raw_data is not None:
    data, newest_date, oldest_date, spending_cols = preprocess_data(raw_data)

    # --- Data Inspection ---
    st.header("1. Data Inspection and Cleaning")
    with st.expander("Show Initial Data Overview"):
        st.subheader("Dataset Head")
        st.dataframe(raw_data.head()) # Show head of the raw data
        st.subheader("Dataset Info")
        buffer = io.StringIO()
        raw_data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.subheader("Dataset Description (Numerical)")
        st.dataframe(raw_data.describe())
        st.subheader("Missing Values per Column")
        st.write(raw_data.isnull().sum())
        if newest_date != "N/A":
             st.write(f"Newest customer enrolment date: {newest_date.strftime('%d-%m-%Y')}")
             st.write(f"Oldest customer enrolment date: {oldest_date.strftime('%d-%m-%Y')}")


    with st.expander("Show Processed Data Sample"):
         st.subheader("Processed Data Head")
         st.dataframe(data.head())
         st.subheader("Processed Data Info")
         buffer_proc = io.StringIO()
         data.info(buf=buffer_proc)
         st.text(buffer_proc.getvalue())


    # --- Exploratory Data Analysis (EDA) ---
    st.header("2. Exploratory Data Analysis (EDA)")
    eda_cols = st.columns(2) # Create two columns for plots

    # Define numerical features based on processed data
    num_features_for_eda = []
    if 'Income' in data.columns: num_features_for_eda.append('Income')
    if 'Age' in data.columns: num_features_for_eda.append('Age')
    if 'Total_Spending' in data.columns: num_features_for_eda.append('Total_Spending')
    if 'Children' in data.columns: num_features_for_eda.append('Children')

    with eda_cols[0]:
        st.subheader("Distributions")
        if 'Income' in data.columns:
            fig1, ax1 = plt.subplots()
            sns.histplot(data['Income'], kde=True, color='skyblue', ax=ax1)
            ax1.set_title("Income Distribution")
            st.pyplot(fig1)
            plt.clf() # Clear the figure

        if 'Age' in data.columns:
            fig2, ax2 = plt.subplots()
            sns.histplot(data['Age'], kde=True, color='salmon', ax=ax2)
            ax2.set_title("Age Distribution")
            st.pyplot(fig2)
            plt.clf()

        if 'Total_Spending' in data.columns:
            fig3, ax3 = plt.subplots()
            sns.histplot(data['Total_Spending'], kde=True, color='gold', ax=ax3)
            ax3.set_title("Total Spending Distribution")
            st.pyplot(fig3)
            plt.clf()

    with eda_cols[1]:
        st.subheader("Relationships")
        if 'Marital_Group' in data.columns and 'Total_Spending' in data.columns:
            fig4, ax4 = plt.subplots(figsize=(7,5)) # Adjust size for column
            avg_spending_by_group = data.groupby('Marital_Group')['Total_Spending'].mean().sort_values(ascending=False)
            sns.barplot(x=avg_spending_by_group.index, y=avg_spending_by_group.values, palette='viridis', ax=ax4)
            ax4.set_title("Avg Total Spending by Marital Group")
            ax4.set_xlabel("Marital Group")
            ax4.set_ylabel("Average Total Spending")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)
            plt.clf()

        # Correlation Heatmap
        if len(num_features_for_eda) > 1: # Need at least 2 numerical features for correlation
            st.subheader("Correlation Heatmap")
            fig5, ax5 = plt.subplots(figsize=(8, 6))
            corr_features = num_features_for_eda + spending_cols # Use original spending columns if available
            corr_features = [f for f in corr_features if f in data.columns] # Ensure columns exist
            if len(corr_features) > 1:
                sns.heatmap(data[corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax5, annot_kws={"size": 8})
                ax5.set_title("Correlation Heatmap of Numerical Features")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig5)
                plt.clf()
            else:
                st.write("Not enough numerical features for a correlation heatmap.")

    # Pair Plot (optional, can be large)
    if st.checkbox("Show Pair Plot (can be slow)", False):
        if len(num_features_for_eda) > 1:
            st.subheader("Pair Plot of Key Numerical Features")
            pair_plot_fig = sns.pairplot(data[num_features_for_eda], diag_kind='kde', palette='coolwarm')
            pair_plot_fig.fig.suptitle("Pair Plot of Key Numerical Features", y=1.02)
            st.pyplot(pair_plot_fig)
            plt.clf()
        else:
            st.write("Not enough numerical features for a pair plot.")


    # --- Clustering ---
    st.header("3. Customer Segmentation using Clustering")

    # Define features for clustering (Make sure these columns exist after preprocessing)
    default_clustering_features = ['Income', 'Age', 'Total_Spending']
    available_features = [f for f in default_clustering_features if f in data.columns]

    if not available_features:
        st.error("Clustering cannot proceed. Required features ('Income', 'Age', 'Total_Spending') not found in the processed data.")
    else:
        clustering_features = st.multiselect(
            "Select features for clustering:",
            options=[col for col in data.select_dtypes(include=np.number).columns if col not in ['Response']], # Allow selecting any numeric column except target
            default=available_features
        )

        if not clustering_features:
             st.warning("Please select at least one feature for clustering.")
        else:
            X_orig = data[clustering_features]
            scaler = StandardScaler()
            X = scaler.fit_transform(X_orig)

            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            data['PCA1'] = X_pca[:, 0]
            data['PCA2'] = X_pca[:, 1]
            st.write(f"Applied PCA for visualization (Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.2f})")

            # --- Sidebar controls for Clustering ---
            st.sidebar.header("Clustering Parameters")
            k_range = st.sidebar.slider("Select Range for K (K-Means, Agglo, GMM):", min_value=2, max_value=10, value=(2, 8))
            eps_options = st.sidebar.text_input("Enter Eps values for DBSCAN (comma-separated):", value="0.5, 0.7, 0.9, 1.1")
            try:
                eps_values = [float(e.strip()) for e in eps_options.split(',')]
            except:
                st.sidebar.error("Invalid Eps values. Please enter comma-separated numbers.")
                eps_values = [0.5, 0.7] # Fallback

            k_values = range(k_range[0], k_range[1] + 1)

            # --- Clustering Models ---
            cluster_tabs = st.tabs(["K-Means", "Agglomerative", "DBSCAN", "GMM"])

            # --- K-Means ---
            with cluster_tabs[0]:
                st.subheader("K-Means Clustering")
                sse = []
                silhouette_scores_km = []
                best_score_km = -1
                best_k_km = k_values[0]

                with st.spinner("Running K-Means for different k..."):
                    for k in k_values:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(X)
                        sse.append(km.inertia_)
                        score = silhouette_score(X, labels)
                        silhouette_scores_km.append(score)
                        if score > best_score_km:
                            best_score_km = score
                            best_k_km = k

                # Plot Elbow Curve
                fig_elbow, ax_elbow = plt.subplots(figsize=(8,4))
                ax_elbow.plot(k_values, sse, marker='o')
                ax_elbow.set_title("Elbow Curve for K-Means")
                ax_elbow.set_xlabel("Number of Clusters (k)")
                ax_elbow.set_ylabel("Sum of Squared Errors (SSE)")
                st.pyplot(fig_elbow)
                plt.clf()

                # Plot Silhouette Scores
                fig_sil_km, ax_sil_km = plt.subplots(figsize=(8,4))
                ax_sil_km.plot(k_values, silhouette_scores_km, marker='s', color='red')
                ax_sil_km.set_title("Silhouette Scores for K-Means")
                ax_sil_km.set_xlabel("Number of Clusters (k)")
                ax_sil_km.set_ylabel("Silhouette Score")
                ax_sil_km.axvline(x=best_k_km, color='blue', linestyle='--', label=f'Best k = {best_k_km} (Score: {best_score_km:.3f})')
                ax_sil_km.legend()
                st.pyplot(fig_sil_km)
                plt.clf()

                st.write(f"**Best K for K-Means based on silhouette score: {best_k_km} (Score: {best_score_km:.3f})**")

                # Final K-Means model and plot
                kmeans_best = KMeans(n_clusters=best_k_km, random_state=42, n_init=10)
                data['Cluster_KMeans'] = kmeans_best.fit_predict(X)
                centers = kmeans_best.cluster_centers_
                centers_pca = pca.transform(centers) # Transform centers to PCA space

                fig_km_pca, ax_km_pca = plt.subplots(figsize=(8,6))
                sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster_KMeans', palette='coolwarm', alpha=0.7, ax=ax_km_pca)
                ax_km_pca.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='black', marker='X', label='Centers')
                ax_km_pca.set_title(f"K-Means Clusters via PCA (k={best_k_km})")
                ax_km_pca.legend()
                st.pyplot(fig_km_pca)
                plt.clf()

                # Optional: Show plots for each k
                if st.checkbox("Show K-Means PCA plots for each k value", False):
                    st.markdown("---")
                    for k in k_values:
                        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels_temp = km_temp.fit_predict(X)
                        silhouette_temp = silhouette_score(X, labels_temp)
                        data['Temp_KMeans'] = labels_temp
                        centers_temp = km_temp.cluster_centers_
                        centers_pca_temp = pca.transform(centers_temp)

                        fig_temp, ax_temp = plt.subplots(figsize=(7,5))
                        sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Temp_KMeans', palette='tab10', alpha=0.7, ax=ax_temp)
                        ax_temp.scatter(centers_pca_temp[:, 0], centers_pca_temp[:, 1], s=150, c='black', marker='X', label='Centers')
                        ax_temp.set_title(f"K-Means PCA Plot for k={k} | Silhouette: {silhouette_temp:.3f}")
                        ax_temp.legend()
                        st.pyplot(fig_temp)
                        plt.clf()
                    data.drop('Temp_KMeans', axis=1, inplace=True) # Clean up temp column
                    st.markdown("---")


            # --- Agglomerative Clustering ---
            with cluster_tabs[1]:
                st.subheader("Agglomerative Clustering")
                agglo_scores = {}
                with st.spinner("Running Agglomerative Clustering for different k..."):
                    for k in k_values:
                        agglo = AgglomerativeClustering(n_clusters=k)
                        labels = agglo.fit_predict(X)
                        # Avoid scoring single cluster results which can happen for k=1 (though range starts at 2)
                        if len(set(labels)) > 1:
                            score = silhouette_score(X, labels)
                            agglo_scores[k] = score
                        else:
                            agglo_scores[k] = -1 # Invalid score

                if agglo_scores:
                    best_k_agglo = max(agglo_scores, key=agglo_scores.get)
                    best_score_agglo = agglo_scores[best_k_agglo]
                    st.write("Silhouette Scores per k:", agglo_scores)
                    st.write(f"**Best K for Agglomerative Clustering: {best_k_agglo} (Score: {best_score_agglo:.3f})**")

                    # Final Agglomerative model and plot
                    agglo_best = AgglomerativeClustering(n_clusters=best_k_agglo)
                    data['Cluster_Agg'] = agglo_best.fit_predict(X)

                    fig_agg_pca, ax_agg_pca = plt.subplots(figsize=(8,6))
                    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster_Agg', palette='viridis', alpha=0.7, ax=ax_agg_pca)
                    ax_agg_pca.set_title(f"Agglomerative Clusters via PCA (k={best_k_agglo})")
                    st.pyplot(fig_agg_pca)
                    plt.clf()

                    # Dendrogram (optional)
                    if st.checkbox("Show Dendrogram for Agglomerative Clustering", False):
                        st.write("Generating Dendrogram (may take a moment)...")
                        fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6))
                        linkage_matrix = sch.linkage(X, method='ward')
                        sch.dendrogram(linkage_matrix, ax=ax_dendro)
                        ax_dendro.set_title('Dendrogram (Ward Linkage)')
                        ax_dendro.set_xlabel('Samples (or cluster index)')
                        ax_dendro.set_ylabel('Distance')
                        # Optionally truncate
                        # sch.dendrogram(linkage_matrix, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, ax=ax_dendro)
                        st.pyplot(fig_dendro)
                        plt.clf()
                else:
                    st.warning("Could not calculate silhouette scores for Agglomerative Clustering.")

            # --- DBSCAN Clustering ---
            with cluster_tabs[2]:
                st.subheader("DBSCAN Clustering")
                dbscan_scores = {}
                results_dbscan = {} # To store labels for plotting

                with st.spinner("Running DBSCAN for different eps values..."):
                     for eps in eps_values:
                        dbscan = DBSCAN(eps=eps, min_samples=5) # min_samples=5 is common default
                        labels = dbscan.fit_predict(X)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Count clusters excluding noise (-1)
                        n_noise = list(labels).count(-1)

                        if n_clusters > 1: # Need at least 2 clusters for silhouette score
                            score = silhouette_score(X[labels != -1], labels[labels != -1]) # Score only non-noise points
                            dbscan_scores[eps] = score
                            st.write(f"DBSCAN eps={eps}: Found {n_clusters} clusters, {n_noise} noise points. Silhouette Score: {score:.3f}")
                        else:
                            st.write(f"DBSCAN eps={eps}: Found {n_clusters} clusters, {n_noise} noise points. (Not enough clusters for Silhouette Score)")
                            dbscan_scores[eps] = -1 # Indicate invalid score

                        results_dbscan[eps] = labels # Store labels

                if any(score > -1 for score in dbscan_scores.values()): # Check if any valid scores were found
                    best_eps_dbscan = max(dbscan_scores, key=dbscan_scores.get)
                    best_score_dbscan = dbscan_scores[best_eps_dbscan]
                    st.write(f"**Best eps for DBSCAN: {best_eps_dbscan} (Score: {best_score_dbscan:.3f})**")
                    data['Cluster_DBSCAN'] = results_dbscan[best_eps_dbscan]
                else:
                    st.warning("DBSCAN did not yield sufficient clusters for evaluation with the chosen parameters.")
                    best_eps_dbscan = eps_values[0] # Default to first eps for plotting
                    data['Cluster_DBSCAN'] = results_dbscan[best_eps_dbscan]

                # Visualization for DBSCAN clusters via PCA (using best eps or default)
                fig_db_pca, ax_db_pca = plt.subplots(figsize=(8,6))
                sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster_DBSCAN', palette='Set1', alpha=0.7, ax=ax_db_pca)
                n_clusters_plot = len(set(data['Cluster_DBSCAN'])) - (1 if -1 in data['Cluster_DBSCAN'].unique() else 0)
                ax_db_pca.set_title(f"DBSCAN Clusters via PCA (eps={best_eps_dbscan}, Clusters: {n_clusters_plot})")
                st.pyplot(fig_db_pca)
                plt.clf()

                # Bar plot showing DBSCAN cluster distribution
                fig_db_count, ax_db_count = plt.subplots(figsize=(8,4))
                sns.countplot(x='Cluster_DBSCAN', data=data, palette='Set2', order = sorted(data['Cluster_DBSCAN'].unique()), ax=ax_db_count)
                ax_db_count.set_title("Distribution of DBSCAN Clusters (-1 = Noise)")
                st.pyplot(fig_db_count)
                plt.clf()

            # --- Gaussian Mixture Model (GMM) ---
            with cluster_tabs[3]:
                st.subheader("Gaussian Mixture Model (GMM) Clustering")
                gmm_scores = {}
                bic_scores = [] # Bayesian Information Criterion (alternative metric)

                with st.spinner("Running GMM for different numbers of components..."):
                    for n in k_values:
                        gmm = GaussianMixture(n_components=n, random_state=42, n_init=5) # n_init added
                        gmm_labels = gmm.fit_predict(X)
                        bic_scores.append(gmm.bic(X)) # Calculate BIC
                        if len(set(gmm_labels)) > 1:
                            score = silhouette_score(X, gmm_labels)
                            gmm_scores[n] = score
                        else:
                             gmm_scores[n] = -1 # Invalid score

                if gmm_scores:
                    best_n_gmm = max(gmm_scores, key=gmm_scores.get)
                    best_score_gmm = gmm_scores[best_n_gmm]
                    st.write("Silhouette Scores per n_components:", gmm_scores)
                    st.write(f"**Best number of components for GMM (Silhouette): {best_n_gmm} (Score: {best_score_gmm:.3f})**")

                    # Plot BIC Scores
                    fig_bic_gmm, ax_bic_gmm = plt.subplots(figsize=(8,4))
                    ax_bic_gmm.plot(k_values, bic_scores, marker='o', color='purple')
                    ax_bic_gmm.set_title("BIC Scores for GMM")
                    ax_bic_gmm.set_xlabel("Number of Components (n)")
                    ax_bic_gmm.set_ylabel("Bayesian Information Criterion (BIC)")
                    best_n_bic = k_values[np.argmin(bic_scores)] # Best n based on BIC is the minimum
                    ax_bic_gmm.axvline(x=best_n_bic, color='orange', linestyle='--', label=f'Best n (BIC) = {best_n_bic}')
                    ax_bic_gmm.legend()
                    st.pyplot(fig_bic_gmm)
                    plt.clf()
                    st.write(f"(Alternative) Best n based on BIC: {best_n_bic}")


                    # Final GMM model and plot (using Silhouette best n)
                    gmm_best = GaussianMixture(n_components=best_n_gmm, random_state=42, n_init=5)
                    data['Cluster_GMM'] = gmm_best.fit_predict(X)

                    fig_gmm_pca, ax_gmm_pca = plt.subplots(figsize=(8,6))
                    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster_GMM', palette='magma', alpha=0.7, ax=ax_gmm_pca)
                    ax_gmm_pca.set_title(f"GMM Clusters via PCA (n={best_n_gmm})")
                    st.pyplot(fig_gmm_pca)
                    plt.clf()

                    # Bar plot showing GMM cluster distribution
                    fig_gmm_count, ax_gmm_count = plt.subplots(figsize=(8,4))
                    sns.countplot(x='Cluster_GMM', data=data, palette='magma', order = sorted(data['Cluster_GMM'].unique()), ax=ax_gmm_count)
                    ax_gmm_count.set_title("Distribution of GMM Clusters")
                    st.pyplot(fig_gmm_count)
                    plt.clf()
                else:
                    st.warning("Could not calculate silhouette scores for GMM.")

    # --- Supervised Learning (Predicting Response) ---
    st.header("4. Predicting Campaign Response (Supervised Learning)")

    if 'Response' not in data.columns:
        st.warning("Target variable 'Response' not found in the data. Skipping supervised learning.")
    else:
        # Prepare data for supervised learning
        # Allow selection of features, default to the ones used for clustering
        sl_features = st.multiselect(
            "Select features for predicting 'Response':",
            options=[col for col in data.select_dtypes(include=np.number).columns if col not in ['Response', 'PCA1', 'PCA2'] and not col.startswith('Cluster_')],
            default=available_features # Use the same available features as clustering default
        )

        if not sl_features:
            st.warning("Please select at least one feature for supervised learning.")
        else:
            X_supervised = data[sl_features]
            y_supervised = data['Response']

            # Check if target variable has variance
            if y_supervised.nunique() < 2:
                st.error("Target variable 'Response' has only one unique value. Cannot train classifier.")
            else:
                # Split the data
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_supervised, y_supervised,
                        test_size=0.3, random_state=42, stratify=y_supervised
                    )
                    st.write(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples.")

                    # Train the Random Forest Classifier
                    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Added class_weight
                    with st.spinner("Training Random Forest Classifier..."):
                        rf.fit(X_train, y_train)

                    # Predictions and evaluation
                    y_pred = rf.predict(X_test)
                    y_proba = rf.predict_proba(X_test)[:, 1]

                    st.subheader("Random Forest Classifier Results")
                    st.text("Classification Report:")
                    # Use monospace font for better alignment
                    st.code(classification_report(y_test, y_pred), language=None)

                    st.text("Confusion Matrix:")
                    st.dataframe(confusion_matrix(y_test, y_pred))

                    roc_auc = roc_auc_score(y_test, y_proba)
                    st.write(f"**ROC AUC Score: {roc_auc:.3f}**")

                    # --- Visualizations ---
                    sl_viz_cols = st.columns(2)

                    with sl_viz_cols[0]:
                        # ROC Curve Visualization
                        st.subheader("ROC Curve")
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        fig_roc, ax_roc = plt.subplots(figsize=(7,6))
                        ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                        ax_roc.plot([0,1], [0,1], 'k--')
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title("ROC Curve for Random Forest")
                        ax_roc.legend(loc='lower right')
                        st.pyplot(fig_roc)
                        plt.clf()

                    with sl_viz_cols[1]:
                        # Feature Importance Visualization
                        st.subheader("Feature Importance")
                        importances = rf.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        features_sl = X_supervised.columns

                        fig_imp, ax_imp = plt.subplots(figsize=(7,6))
                        sns.barplot(x=importances[indices], y=features_sl[indices], palette='viridis', ax=ax_imp)
                        ax_imp.set_title("Feature Importance from Random Forest")
                        ax_imp.set_xlabel("Importance Score")
                        ax_imp.set_ylabel("Features")
                        st.pyplot(fig_imp)
                        plt.clf()

                except Exception as e:
                    st.error(f"An error occurred during supervised learning: {e}")
                    st.info("This might happen if the 'Response' variable distribution is highly skewed after splitting.")


    st.markdown("---")
    st.write("Analysis Complete.")

else:
    st.info("Please upload an Excel file using the sidebar to begin the analysis.")
