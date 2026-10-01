from sklearn.linear_model import LinearRegression
import joblib
import os


def train_model(X_train, y_train):
    """
    Entraîne un modèle de régression linéaire.

    Paramètres
    ----------
    X_train : pd.DataFrame
        Features d'entraînement
    y_train : pd.Series
        Target d'entraînement

    Retour
    ------
    model : LinearRegression
        Modèle entraîné
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def save_model(model, model_path: str):
    """
    Sauvegarde le modèle entraîné dans un fichier .pkl.

    Paramètres
    ----------
    model : modèle sklearn
        Modèle entraîné
    model_path : str
        Chemin de sauvegarde (ex: models/linear_regression.pkl)
    """

    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)


def load_model(model_path: str):
    """
    Charge un modèle sauvegardé.

    Paramètres
    ----------
    model_path : str
        Chemin du fichier .pkl

    Retour
    ------
    model : modèle sklearn
        Modèle chargé
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le fichier modèle {model_path} est introuvable."
        )

    model = joblib.load(model_path)
    return model
