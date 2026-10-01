import pandas as pd


def preprocess_data(df: pd.DataFrame):
    """
    Nettoie et prépare les données pour la modélisation.

    Étapes :
    - suppression des colonnes inutiles
    - séparation features / target
    - encodage des variables catégorielles yes/no
    """

    # Copie pour éviter les effets de bord
    df = df.copy()

    # Suppression de la colonne inutile
    if "rownames" in df.columns:
        df.drop(columns=["rownames"], inplace=True)

    # Séparation target / features
    if "price" not in df.columns:
        raise ValueError("La colonne 'price' est absente du dataset.")

    y = df["price"]
    X = df.drop(columns=["price"])

    # Détection des colonnes catégorielles
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Encodage yes / no → 1 / 0
    for col in cat_cols:
        X[col] = X[col].map({"yes": 1, "no": 0})

    return X, y
