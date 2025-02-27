import streamlit as st
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

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

def encoding_dataset(df):
    """Transforme les colonnes categoriques en colonnes numeriques"""

    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder() # On cree un LabelEncoder pour chaque colonne pour eviter de creer une dependance entre les encoder
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded


def handle_missing_values(df, method):
    """Complete les valeurs manquantes selon la methode choisie"""

    df_handled = df.copy()
    #if method == "Clustering":

    #elif method == "Mean":

    #elif method == "Median":
    
    return df_handled


def important_features(df, cible):
    """Garde uniquement les k colonnes les plus importantes selon la cible (k etant optimal)"""

    X = df.drop(columns=[cible])
    y = df[cible]
    best_k = 0
    best_score = 0
    problem_type = "Classification"

    for val in y.shape[1]:
        if val != 0 and val != 1:
            problem_type = "Regression"

    for k in range(1, X.shape[1] + 1): # choix du k le plus optimal
        if problem_type == "Classification":
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            
            model = RandomForestClassifier()
            model.fit(X_selected, y)
        
            y_pred = model.predict_proba(X_selected)[:, 1]
            score = roc_auc_score(y, y_pred)  

        else :
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            
            model = LinearRegression()
            score = cross_val_score(model, X_selected, y, cv=5, scoring='r2').mean()
        
        if score > best_score:
            best_k = k
            best_score = score

    selector = SelectKBest(score_func=f_classif, k=best_k)
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    df_important_features = X[selected_columns]

    return df_important_features


def traitement_df(df, cible):
    """Effectue le traitement necessaire sur le df"""

    df_encoded = encoding_dataset(df)
    df_handled = handle_missing_values(df_encoded)
    df_important_features = important_features(df_handled, cible)

    return df_important_features


#MODELE DE MACHINE LEARNING

def regression(df, cible):
    """Predit une valeur numerique pour la colonne cible"""

def classification(df, cible):
    """Predit une categorie pour la colonne cible"""

def classification_image(df):
    """Donne une probabilite de classe pour une certaine image en input"""


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
        method = st.selectbox("Methods", ["Clustering", "Mean", "Median"])
        if st.button("Traitement du Dataset"):
            traitement_df(df, cible)

        st.write(df.dtypes)
    else:
        st.warning("No file selected. Please upload a CSV.")

elif page == "Data Visualization":
    st.title("Data Visualization")
    

elif page == "Data Prediction":
    st.title("Data Prediction")