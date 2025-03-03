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
from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error, r2_score, accuracy_score, silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from  xgboost import XGBRegressor


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
        if df_clean[col].astype(str).str.contains("\$").any(): # FIXME faire pour toutes les colonnes si jamais le prix n'est pas notre colonne cible
            df_clean[col] = df_clean[col].str.replace("$", "", regex=False) 
            df_clean[col] = df_clean[col].str.replace(",", "", regex=False)  
            df_clean[col] = pd.to_numeric(df_clean[col])  

        if df_clean[col].astype(str).str.contains(r"mi\.").any(): # FIXME faire pour toutes les colonnes si jamais le prix n'est pas notre colonne cible
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
                      if df_encoded[col].nunique() > 10]
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder() # On cree un LabelEncoder pour chaque colonne pour eviter de creer une dependance entre les encoder
        df_encoded[col] = le.fit_transform(df_encoded[col])

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
            
            model = LinearRegression()
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

    y_pred = model.predict(X_test)

    return(y_test, y_pred)


def classification_image(df):
    """Donne une probabilite de classe pour une certaine image en input"""
    # TODO


# DATA VISUALIZATION 

def plot_predictions(y_test, y_pred):
    """Affiche un plan, si les points son proche de y=x alors le modèle a une bonne prédiction"""

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Prédictions")
    ax.set_title("Comparaison entre les vraies valeurs et les prédictions")
    
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
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.histplot(df[colonne], kde=True, ax=ax)
        g_col.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(3, 2))
        feature_counts = df[colonne].value_counts()
        ax2.pie(feature_counts, labels=feature_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"Répartition de : {colonne}")
        d_col.pyplot(fig2)


# AFFICHAGE STREAMLIT

page = st.sidebar.radio("Navigation", ["Homepage", "Data Infos", "Data Prediction", "Data Visualization"])
uploaded_file = st.sidebar.file_uploader("Import CSV file", type=["csv"])

# Gere les differentes erreurs de chargement du fichier 
if uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded_file)  # Preserve l'affichage du CSV quand on change de page
        st.sidebar.success("The file has been uploaded")
    except Exception as e:
        st.sidebar.error(f"❌ Loading error : {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
else:
    df = None


if page == "Homepage":
    st.title("Homepage")

elif page == "Data Infos":
    st.title("Data Infos")
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


    else:
        st.warning("No file selected. Please upload a CSV.")

elif page == "Data Prediction":
    st.title("Data Prediction")

    if df is not None:
        df = remove_unit(df)
        cible = st.selectbox("Column to predict", df.columns)
        method = st.selectbox("Methods", ["Frequency/Mean", "Clustering"])
        problem_type = determine_problem_type(df, cible)

        if 'df_preprocessed' and 'preprocessed' not in st.session_state:
            # Garde en mémoire le traitement du dataset même si on clique plus sur le bouton 
            st.session_state.df_preprocessed = pd.DataFrame()
            st.session_state.preprocessed = False
        if "y_test" not in st.session_state:
            st.session_state.y_test = np.array([])  # Tableau vide

        if "y_pred" not in st.session_state:
            st.session_state.y_pred = np.array([])

        if st.button("Traitement du Dataset"):
            if df is not None and cible is not None and method is not None:
                try:
                    st.session_state.df_preprocessed = traitement_df(df, cible, method)
                    st.success("The dataset has been successfully processed.")
                    st.write(f"**Total of missing values :** {st.session_state.df_preprocessed.isna().sum().sum()}")
                    st.session_state.preprocessed = True
                except Exception as e:
                    st.warning(f"An error occurred: {str(e)}")
            else:
                st.warning("Please ensure that the dataset, target column, and method are selected.")
        elif not st.session_state.preprocessed:
            st.warning("Please preprocess your file by clicking on the button.") 

        if st.button("Predict the target column"):
            if st.session_state.df_preprocessed.empty:
                st.write('The dataset is empty')
            else:
                st.header("Prediction Report:")
                sel_col, disp_col = st.columns(2)

                if problem_type == "Classification": 
                    st.session_state.y_test, st.session_state.y_pred = classification(st.session_state.df_preprocessed, cible)

                    disp_col.subheader("Precision score of the model")
                    disp_col.metric("Precision : ", f"{accuracy_score(st.session_state.y_test, st.session_state.y_pred): .2f}")

                elif problem_type == "Regression":
                    st.session_state.y_test, st.session_state.y_pred = regression(st.session_state.df_preprocessed, cible)

                    disp_col.subheader("Mean squared error of the model")
                    disp_col.metric("MSE : ", f"{mean_squared_error(st.session_state.y_test, st.session_state.y_pred): .4f}")
                    disp_col.subheader("R² score of the model")
                    disp_col.metric("R² score : ", f"{r2_score(st.session_state.y_test, st.session_state.y_pred): .4f}")
    else:
        st.warning("No file selected. Please upload a CSV.")

elif page == "Data Visualization":
    st.title("Data Visualization")
    plot_predictions(st.session_state.y_test, st.session_state.y_pred)
    heat_map(st.session_state.df_preprocessed)