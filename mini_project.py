import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error, r2_score, accuracy_score, silhouette_score, confusion_matrix, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from  xgboost import XGBRegressor
from statsmodels.graphics.mosaicplot import mosaic


st.set_page_config(
    page_title = "Mini-project application",
    layout = "wide"
)


# TRAITEMENT DU DATASET

def determine_problem_type(df, cible, threshold=10):
    """Détermine si la colonne cible correspond à une classification ou une régression"""

    problem_type = "Regression"
    unique_values = df[cible].nunique()
    
    if unique_values <= threshold:
        problem_type = "Classification"

    return problem_type


def remove_unit(df):
    """Retire le $ dans les colonnes de prix pour bien que la colonne soit numérique"""

    df_clean = df.copy()
    for col in df.columns :
        #retire les $ pour avoir le prix numérique et non en catégorie
        if df_clean[col].astype(str).str.contains("\$").any():
            df_clean[col] = df_clean[col].str.replace("$", "", regex=False) 
            df_clean[col] = df_clean[col].str.replace(",", "", regex=False)  
            df_clean[col] = pd.to_numeric(df_clean[col])  

        if df_clean[col].astype(str).str.contains(r"mi\.").any():
            df_clean[col] = df_clean[col].str.replace("mi.", "", regex=False) 
            df_clean[col] = df_clean[col].str.replace(",", "", regex=False)  
            df_clean[col] = pd.to_numeric(df_clean[col]) 

    return df_clean

def remove_extreme_values(df, threshold = 1.5):
    """Retire les valeurs aberrantes du dataset"""

    df_wout_xtr_values = df.copy()
    num_cols = [ col for col in df_wout_xtr_values.select_dtypes(include=['int64', 'float64']).columns if df_wout_xtr_values[col].nunique() >10]

    for col in num_cols:
        Q1 = df_wout_xtr_values[col].quantile(0.25)
        Q3 = df_wout_xtr_values[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Définition des bornes
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filtrer les valeurs aberrantes
        df_wout_xtr_values = df_wout_xtr_values[(df_wout_xtr_values[col] >= lower_bound) & (df_wout_xtr_values[col] <= upper_bound)]

    return df_wout_xtr_values


def encoding_dataset(df):
    """Transforme les colonnes categoriques en colonnes numeriques"""

    df_encoded = df.copy()
    numerical_cols = [col for col in df_encoded.select_dtypes(include=['int64', 'float64']).columns 
                      if df_encoded[col].nunique() > 10 or df_encoded[col].nunique() == 1]
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    st.session_state["encoders"] = {}
    if "encoders" in st.session_state:
        encoders = st.session_state["encoders"]

    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder() # On cree un LabelEncoder pour chaque colonne pour eviter de creer une dependance entre les encoder
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded


def handle_missing_values(df, method, cible):
    """Complete les valeurs manquantes selon la methode choisie"""

    df_handled = df.copy()
    if method == "Frequency/Mean":
        df_handled = df_handled.apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col 
                                      if col.dtypes == 'O' else col)
        df_handled = df_handled.apply(lambda col: col.fillna(col.mean()) if col.dtypes != 'O' else col)
    
    elif method == "Clustering":
        c = 5 # Nombre de colonne de corrélation utilisées pour faire les clusters
        max_k = 10 # Nombre maximum de clusters

        corr_matrix = df.corr().abs() 
        cible_corr = corr_matrix[cible].drop(cible).sort_values(ascending=False)  
        best_cols = cible_corr.index[:c] 

        df_cluster = df_handled[best_cols].dropna()

        distortions = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)  # Tester de 2 à max_k clusters
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_cluster)
            
            distortions.append(sum(np.min(cdist(df_cluster, kmeans.cluster_centers_, 'euclidean'), axis=1)) / df_cluster.shape[0])
            silhouette_scores.append(silhouette_score(df_cluster, kmeans.labels_))

        # Trouver le k optimal (min du coude de la courbe d’inertie)
        optimal_k = K_range[np.argmax(silhouette_scores)]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(K_range, distortions, marker='o', linestyle='-') # FIXME A retirer ou à déplacer dans la page de data prediciton
        ax[0].set_title("Méthode du coude (Inertie)")
        ax[0].set_xlabel("Nombre de clusters")
        ax[0].set_ylabel("Distorsion")

        # Score de silhouette
        ax[1].plot(K_range, silhouette_scores, marker='o', linestyle='-', color='red') # FIXME A retirer ou à déplacer dans la page de data prediciton
        ax[1].set_title("Score de silhouette")
        ax[1].set_xlabel("Nombre de clusters")
        ax[1].set_ylabel("Silhouette Score")

        st.pyplot(fig) 

        st.write(f"### Nombre optimal de clusters choisi : {optimal_k}")

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_handled["Cluster"] = kmeans.fit_predict(df_cluster)

        for col in df_handled.columns:
            if df_handled[col].isna().sum() > 0:  # Si la colonne contient des NaN
                if df_handled[col].dtype in ['int64', 'float64']: 
                    df_handled[col] = df_handled.groupby("Cluster")[col].transform(lambda x: x.fillna(x.median()))


        df_handled.drop(columns=["Cluster"], inplace=True)
    # FIXME Ne marche pas pour toutes les colonnes (dans stroke dataset voir 'age')
    # TODO elif method == "Median":
    
    return df_handled


def important_features(df, cible):
    """Garde uniquement les k colonnes les plus importantes selon la cible (k etant optimal)"""

    X = df.drop(columns=[cible])
    y = df[cible]
    best_k = 0
    best_score = 0
    problem_type = determine_problem_type(df, cible)

    for k in range(1, X.shape[1] + 1): # choix du k le plus optimal
        if problem_type == "Classification":
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            
            model = RandomForestClassifier()
            model.fit(X_selected, y)

            y_pred_proba = model.predict_proba(X_selected)

            if y_pred_proba.shape[1] == 2:
                y_pred = y_pred_proba[:, 1]  # Prendre la proba de la classe positive
            else:
                y_pred = y_pred_proba[:, 0]
            score = roc_auc_score(y, y_pred)  

        elif problem_type == "Regression" :
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            
            model = LinearRegression() # TODO tester avec un XGBoost
            score = cross_val_score(model, X_selected, y, cv=5, scoring='r2').mean()
        
        if score > best_score:
            best_k = k
            best_score = score

    if best_k < 3: # Impose d'avoir au moins 3 colonnes de selectionnees
        best_k = 3
    if problem_type == "Classification":
        selector = SelectKBest(score_func=f_classif, k=best_k)
    elif problem_type == "Regression":
        selector = SelectKBest(score_func=f_regression, k=best_k)

    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    df_important_features = X[selected_columns]
    df_important_features.loc[:, cible] = df[cible]

    return df_important_features


def traitement_df(df, cible, method):
    """Effectue le traitement necessaire sur le df"""

    df_wout_xtr_values = remove_extreme_values(df)
    df_encoded = encoding_dataset(df_wout_xtr_values)
    df_handled = handle_missing_values(df_encoded, method, cible)
    df_important_features = important_features(df_handled, cible)
    st.write("### Most important features used to train the model :")
    st.write(df_important_features.head())

    return df_important_features


#MODELES DE MACHINE LEARNING

def regression(df, cible):
    """Predit une valeur numerique pour la colonne cible"""
    
    y = df[cible]
    X = df.drop(columns=[cible])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        objective="reg:squarederror",  # Objectif de régression
        n_estimators=None,  # Nombre d'arbres
        max_depth=6,  # Profondeur maximale des arbres
        learning_rate=0.1,  # Taux d'apprentissage (step size)
        subsample=0.8,  # Sous-échantillonnage des données
        colsample_bytree=0.8,  # Proportion de features utilisées par arbre
        random_state=42,
        verbosity=0  # Réduire les logs de XGBoost
    )

    model.fit(X_train, y_train)

    st.session_state["trained_model"] = model
    st.session_state["feature_columns"] = X.columns
    if "trained_model" in st.session_state and "feature_columns" in st.session_state:
        trained_model = st.session_state["trained_model"]
        feature_columns = st.session_state["feature_columns"] 

    y_pred = model.predict(X_test)
    return(y_test, y_pred)


def classification(df, cible):
    """Predit une categorie pour la colonne cible"""

    y = df[cible]
    X = df.drop(columns=[cible])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, class_weight='balanced') # TODO pouvoir choisir la profondeur du modele
    model.fit(X_train, y_train)

    st.session_state["trained_model"] = model
    st.session_state["feature_columns"] = X.columns
    if "trained_model" in st.session_state and "feature_columns" in st.session_state:
        trained_model = st.session_state["trained_model"]
        feature_columns = st.session_state["feature_columns"] 

    y_pred = model.predict(X_test)

    return(y_test, y_pred)


# DATA VISUALIZATION 

def plot_predictions(y_test, y_pred):
    """Affiche un plan, si les points son proche de y=x alors le modèle a une bonne prédiction"""

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel("Real values")
    ax.set_ylabel("Predictions")
    ax.set_title("Comparison between true values ​​and predictions")
    
    st.pyplot(fig)


def heat_map(df):
    """Affiche une heatmap pour voir les corrélations entre les variables"""

    fig, ax = plt.subplots(figsize=(10,6))  
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)  
    st.pyplot(fig)  


def plot_missing_values(df):
    """Affiche une heatmap et un bar plot des valeurs manquantes dans le dataset"""

    g_col, d_col = st.columns([2, 1])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna(), cmap="viridis", cbar=False, ax=ax)
    g_col.pyplot(fig)

    missing_values = df.isna().sum().sort_values(ascending=False)
    d_col.bar_chart(missing_values[missing_values > 0])


def plot_column(df, colonne):
    """Affiche un histogramme de la colonne ainsi qu'un boxplot pour détecter les valeurs aberrantes"""

    g_col, d_col = st.columns(2)

    if df[colonne].dtype in [np.int64, np.float64]:
        fig, ax = plt.subplots()
        sns.histplot(df[colonne], kde=True, ax=ax)
        g_col.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[colonne], ax=ax2)
        d_col.pyplot(fig2)
    
    else:
        categories = df[colonne]
        if df[colonne].nunique() > 10:
            st.warning("The selected column has too many different categories, only the 10 most frequent are displayed")
            categories = df[colonne].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.barplot(x=categories.index, y=categories.values, ax=ax)
        ax.set_xlabel('Category')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=90, ha='right')
        g_col.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(3, 2))
        feature_counts = categories.value_counts()
        ax2.pie(feature_counts, labels=feature_counts.index, autopct='%1.1f%%', startangle=90) # FIXME problème quand il y a trop de catégories différentes
        ax2.set_title(f"Répartition de : {colonne}")
        d_col.pyplot(fig2)


def plot_correlation(df, colonne):
    """Affiche un scatter plot de corrélation entre les 4 colonnes les plus corrélées à la colonne à prédire """

    g_col, d_col = st.columns(2)
    g_col2, d_col2 = st.columns(2)

    corr = df.corr()[colonne].sort_values(ascending=False)
    top_corr_features = corr.index[1:5]  # Sélectionne les 4 variables les plus corrélées

    for col in top_corr_features:
        i=1
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col], y=df[colonne], ax=ax)
        ax.set_title(f"{col} vs {colonne}")
        if i==1:
            g_col.pyplot(fig)
        elif i==2:
            d_col.pyplot(fig)
        elif i==3:
            g_col2.pyplot(fig)
        elif i==4:
            d_col2.pyplot(fig)
        i+=1


def confusion_mat(y_test, y_pred):
    """Affiche une matrice de confusion pour identifier les faux positifs, etc..."""

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion")

    st.pyplot(fig)


# AFFICHAGE STREAMLIT

page = st.sidebar.radio("Navigation", ["Homepage", "EDA", "Data Prediction", "Data Visualization", "User forms"])
uploaded_file = st.sidebar.file_uploader("Import CSV file", type=["csv"])

# Gere les differentes erreurs de chargement du fichier 
if uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded_file)  # Preserve l'affichage du CSV quand on change de page
        st.sidebar.success("The file has been uploaded")
    except Exception as e:
        st.sidebar.error(f"❌ Loading error : {e}")
else:
    if "df" in st.session_state: # Si le dataset est supprimé par l'utilisateur, on supprime toutes les variables de sessions
        del st.session_state["df"]
    if "cible" in st.session_state:
        del st.session_state["cible"]
    if "y_pred" in st.session_state:
        del st.session_state["y_pred"]
    if "y_test" in st.session_state:
        del st.session_state["y_test"]
    if "df_preprocessed" in st.session_state:
        del st.session_state["df_preprocessed"]
    if "preprocessed" in st.session_state:
        del st.session_state["preprocessed"]
    if "trained_model" in st.session_state:
        del st.session_state["trained_model"]
    if "feature_columns" in st.session_state:
        del st.session_state["feature_columns"]
    if "encoders" in st.session_state:
        del st.session_state["encoders"]
    if "defined_classes" in st.session_state:
        del st.session_state["defined_classes"]

# Initialisation des variables globales et inter-pages

if "df" in st.session_state:
    df = st.session_state["df"]
else:
    df = None

if "cible" in st.session_state:
    cible = st.session_state["cible"]
else:
    cible = None

if 'preprocessed' in st.session_state:
    preprocessed = st.session_state["preprocessed"]
else:
    preprocessed = None

if 'df_preprocessed' in st.session_state:
    df_preprocessed = st.session_state["df_preprocessed"]
else:
    df_preprocessed = None

if "y_test" in st.session_state:
    y_test = st.session_state['y_test']
else:
    y_test = None

if "y_pred" in st.session_state:
    y_pred = st.session_state["y_pred"]
else:
    y_pred = None

if "trained_model" in st.session_state:
    trained_model = st.session_state["trained_model"]
else:
    trained_model = None

if "feature_columns" in st.session_state:
    feature_columns = st.session_state["feature_columns"]
else:
    feature_columns = None

if "encoders" in st.session_state:
    encoders = st.session_state["encoders"]
else:
    encoders = None

if "defined_classes" in st.session_state:
    defined_classes = st.session_state["defined_classes"]
else:
    defined_classes = None

if df is not None:
    df_originel = df.copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Affichage de chaque page

if page == "Homepage":
    st.title("Homepage")
    st.balloons()
    st.markdown("""
    <div style="background-color:#f0f0f5; padding: 10px; border-radius: 10px;">
        <h3>Welcome to our <i>Exploratory Data Analysis App</i></h3>
        <p>In this section, we will explain how each page of our application works.</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("EDA Page")
    with st.expander("See more"):
        st.write("""An Exploratory Data Analysis (EDA) page provides an overview of a dataset, 
            helping analysts uncover patterns, identify outliers, and spot missing values. 
            It presents key statistics and visualizations, such as histograms and scatter plots, 
            to highlight the distribution and relationships between variables. 
            This initial analysis helps in detecting data quality issues, guiding data cleaning and feature engineering. 
            The EDA page is essential for informed decision-making before proceeding with advanced modeling or analysis.""")
    
    st.header("Data Prediction Page")
    with st.expander("See more"):
        st.write("""The Data Prediction page allows users to train a regression 
        or classification model based on a selected target column. Users can choose the column they want to predict,
         and the page will automatically adjust the model type accordingly. It provides options to preprocess the data,
          select features, and evaluate the model's performance using metrics like accuracy or RMSE. 
          This interactive page simplifies the model training process, making it accessible for users to experiment with different algorithms.
           It enables quick insights into model performance and potential improvements.""")
        
    st.header("Data Visualization Page")
    with st.expander("See more"):
        st.write("""The Data Visualization page displays key visualizations to assess the performance of a machine learning model.
         It includes graphs like confusion matrices, ROC curves, and residual plots, depending on whether the model is for classification
          or regression. These visual tools help users understand how well the model is performing and identify areas for improvement. 
          The page allows for easy comparison between different models or hyperparameters. It enhances the analysis by providing
           clear insights into model accuracy, error distribution, and other performance metrics.""")
        
    st.header("User Forms Page")
    with st.expander("See more"):
        st.write("""The User Forms page allows users to input their own data and get predictions from the trained model.
         Based on the target column selected earlier, the page adjusts to predict either a continuous value (regression)
          or a class label (classification). Users can enter values for the chosen features, and the model will output a prediction in real time.
           This page offers an interactive way for users to test how the model performs with new, unseen data. 
           It provides instant feedback, helping users understand how the model generalizes to different inputs.""")        

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    if df is not None:
        df = remove_unit(df)

        st.header("Dataset preview :")
        st.dataframe(df.head())
        g_col, d_col = st.columns(2)
        g_col.header("Descriptive stats :")
        g_col.write(df.describe())
        d_col.header("Column names :")
        d_col.write(df.dtypes)
        st.header("Dataset information :")
        st.markdown(f"""
            - **Number of rows :** {df.shape[0]}  
            - **Number of columns :** {df.shape[1]}  
            - **Total of missing values :** {df.isna().sum().sum()}
            """)

        if df.isna().sum().sum() > 0:
            st.header("Heatmap and Bar plot of missing values :")
            plot_missing_values(df)
        
        colonne = st.selectbox("Column to plot", df.columns)
        st.header("Distribution of the variable :")
        plot_column(df, colonne)

        g_col2, d_col2 = st.columns(2)
        colonne1 = g_col2.selectbox("First column to compare", df.columns)
        colonne2 = g_col2.selectbox("Second column to compare", df.columns)
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.scatterplot(data=df, x=colonne1, y=colonne2, palette="coolwarm", ax=ax)
        ax.set_title(f"Relation between {colonne1} and {colonne2}")
        d_col2.pyplot(fig)

    else:
        st.warning("No file selected. Please upload a CSV.")

elif page == "Data Prediction":
    st.title("Data Prediction")

    if df is not None:
        df = remove_unit(df)

        st.session_state["cible"] = st.selectbox("Column to predict", df.columns)
        if "cible" in st.session_state:
            cible = st.session_state["cible"]
        
        if df[cible].dtype in ["int64", "float64"] and df[cible].nunique()<11:
            st.warning("The target column you choose is already encoded. Please describe the classes.")

            st.session_state["defined_classes"] = {}
            if 'defined_classes' in st.session_state:
                defined_classes = st.session_state["defined_classes"]

            with st.form("Information Request"):
                for cls in df[cible].unique():
                    expl_classe = st.text_input(f"Definition for class '{cls}':") 
                    defined_classes[cls] = expl_classe

                submitted = st.form_submit_button("Submit")
                
                if submitted :
                    st.success("The definitions have been taken into account")

        method = st.selectbox("Methods", ["Frequency/Mean", "Clustering"])
        problem_type = determine_problem_type(df, cible)

        st.session_state["preprocessed"] = False
        if 'preprocessed' in st.session_state:
            preprocessed = st.session_state["preprocessed"]

        if st.button("Preprocess the dataset"):
            if df is not None and cible is not None and method is not None:
                try:
                    st.session_state["df_preprocessed"] = traitement_df(df, cible, method)
                    if 'df_preprocessed' in st.session_state:
                        df_preprocessed = st.session_state["df_preprocessed"]

                    st.success("The dataset has been successfully processed.")
                    st.write(f"**Total of missing values :** {df_preprocessed.isna().sum().sum()}")
                    preprocessed = True
                except Exception as e:
                    st.warning(f"An error occurred: {str(e)}")
            else:
                st.warning("Please ensure that the dataset, target column, and method are selected.")
        elif not preprocessed:
            st.warning("Please preprocess your file by clicking on the button.") 

        if st.button("Predict the target column"):
            if df_preprocessed.empty:
                st.write('The dataset is empty')
            else:
                st.header("Prediction Report:")

                if problem_type == "Classification": 
                    st.session_state["y_test"], st.session_state["y_pred"] = classification(df_preprocessed, cible)

                    if "y_test" in st.session_state:
                        y_test = st.session_state['y_test']

                    if "y_pred" in st.session_state:
                        y_pred = st.session_state["y_pred"]

                    st.subheader("Precision score of the model")
                    st.metric("Precision : ", f"{accuracy_score(y_test, y_pred): .2f}")

                    st.subheader("F1 score of the model")
                    st.metric("F1 score : ", f"{f1_score(y_test, y_pred, average='weighted'): .2f}")

                elif problem_type == "Regression":
                    st.session_state["y_test"], st.session_state["y_pred"] = regression(df_preprocessed, cible)

                    if "y_test" in st.session_state:
                        y_test = st.session_state['y_test']

                    if "y_pred" in st.session_state:
                        y_pred = st.session_state["y_pred"]

                    st.subheader("Mean squared error of the model")
                    st.metric("MSE : ", f"{mean_squared_error(y_test, y_pred): .4f}")
                    st.subheader("R² score of the model")
                    st.metric("R² score : ", f"{r2_score(y_test, y_pred): .4f}")
    else:
        st.warning("No file selected. Please upload a CSV.")

elif page == "Data Visualization":
    st.title("Data Visualization")
    
    if df is not None :
        if df_preprocessed is not None:
            problem_type = determine_problem_type(df, cible)

            if y_test is not None and y_pred is not None:
                plot_predictions(y_test, y_pred)

                st.header("Correlation plot :")
                heat_map(df_preprocessed)
                plot_correlation(df_preprocessed, cible)

                if problem_type == "Classification":
                    confusion_mat(y_test, y_pred)
            else:
                st.warning("Please press the 'Predict the target column' button on 'Data Prediction' page")
        else:
            st.warning("Please press the 'Preprocess the dataset' button on the 'Data Preprocessing' page")
    else:
        st.warning("No file selected. Please upload a CSV.")

elif page == "User forms":
    st.title("User forms")
    
    if df is not None:
        if cible is not None: 
            st.write(df_originel.head())
            df_originel = remove_unit(df_originel)
            problem_type = determine_problem_type(df_originel, cible)
            col_cible = df_originel[cible]
            df_originel = df_originel.drop(columns=[cible])

            numerical_cols = [col for col in df_originel.select_dtypes(include=['int64', 'float64']).columns if df_originel[col].nunique() >= 3]
            categorical_cols = [col for col in df_originel.select_dtypes(include=['int64', 'float64']).columns if df_originel[col].nunique() == 2]
            selected_cols = [col for col in df_originel.columns if col not in numerical_cols and col not in categorical_cols and df_originel[col].nunique()<11]
            text_cols = [col for col in df_originel.columns if col not in numerical_cols and df_originel[col].nunique()>10]

            user_input = {}
            with st.form("Patient form"):
                for col in df_originel.columns:
                    if col == "id" or col == "ID" or col == "Id":
                        colonne = st.number_input(f"{col}", min_value=df_originel[col].max())

                    elif col in numerical_cols:
                        colonne = st.number_input(f"{col}")

                    elif col in text_cols:
                        colonne = st.text_input(f"{col}")

                    elif col in selected_cols:
                        colonne = st.selectbox(f"{col}", df[col].unique())
                        
                    elif col in categorical_cols:
                        colonne = st.checkbox(f"{col}")

                    user_input[col] = colonne

                submitted = st.form_submit_button("Predict")

                if submitted:
                    if "trained_model" in st.session_state:
                        filtered_input = {key: user_input[key] for key in feature_columns}
                        user_data = pd.DataFrame([filtered_input], columns=feature_columns)
                        user_data.loc[:, user_data.select_dtypes(include=bool).columns] = user_data.select_dtypes(include=bool).astype(int)

                        for col in user_data.select_dtypes(include=["object", "category"]).columns:
                            if col in encoders: 
                                try:
                                    user_data[col] = encoders[col].transform(user_data[col])
                                except:
                                    user_data[col] = -1
                            else:
                                user_data[col] = -1  # Valeur inconnue (cas où une nouvelle catégorie apparaît)

                        prediction = trained_model.predict(user_data)

                        if problem_type == "Classification":
                            if col_cible.dtype in ["int64", "float64"]:
                                st.success(f"The predicted class for {cible} is : **{defined_classes[prediction[0]]}**")

                            else:
                                le_cible = encoders[cible]
                                prediction_label = le_cible.inverse_transform([prediction[0]])[0]
                                st.success(f"The predicted class for {cible} is : **{prediction_label}**")

                        else:
                            cible_mean = col_cible.mean()
                            cible_std = col_cible.std(ddof=0)

                            prediction_destandard = prediction[0] * cible_std + cible_mean

                            st.success(f"Model Prediction : **{prediction_destandard:.2f}**")
                    else:
                        st.warning("Le modèle n'a pas encore été entraîné. Chargez un dataset pour l'entraîner.")
        else:
            st.warning("Please select the column you want to predict on the 'Data Prediction' page")
    else:
        st.warning("No file selected. Please upload a CSV.")
