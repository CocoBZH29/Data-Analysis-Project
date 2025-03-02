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
from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from  xgboost import XGBRegressor

st.set_page_config(
    page_title = "Mini-project application",
    layout = "wide"
)

page = st.sidebar.radio("Navigation", ["Homepage", "Data Infos", "Data Visualization", "Data Prediction"])
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


# TRAITEMENT DU DATASET

def determine_problem_type(df, cible, threshold=10):
    """Détermine si la colonne cible correspond à une classification ou une régression"""

    problem_type = "Regression"
    unique_values = df[cible].nunique()
    
    if unique_values <= threshold:
        problem_type = "Classification"

    return problem_type


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


def handle_missing_values(df, method):
    """Complete les valeurs manquantes selon la methode choisie"""

    df_handled = df.copy()
    if method == "Frequency":
        df_handled = df_handled.apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col 
                                      if col.dtypes == 'O' else col)
    
    df_handled = df_handled.apply(lambda col: col.fillna(col.mean()) if col.dtypes != 'O' else col)

    # TODO elif method == "Mean":

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
        
            y_pred = model.predict_proba(X_selected)[:, 1]
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

    df_encoded = encoding_dataset(df)
    df_handled = handle_missing_values(df_encoded, method)
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


# AFFICHAGE STREAMLIT

if page == "Homepage":
    st.title("Homepage")

elif page == "Data Infos":
    st.title("Data Infos")
    if df is not None:
        st.header("Dataset preview :")
        st.dataframe(df.head())
        st.write(df.describe())
        st.header("Dataset information :")
        st.markdown(f"""
            - **Number of rows :** {df.shape[0]}  
            - **Number of columns :** {df.shape[1]}  
            - **Total of missing values :** {df.isna().sum().sum()}
            """)

        cible = st.selectbox("Column to predict", df.columns)
        method = st.selectbox("Methods", ["Frequency", "Clustering", "Mean", "Median"])
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
                    st.session_state.df_preprocessed = traitement_df(df, cible, method).copy()
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
                st.write('Le dataset est vide')
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
    

elif page == "Data Prediction":
    st.title("Data Prediction")