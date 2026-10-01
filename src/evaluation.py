from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle de régression.

    Paramètres
    ----------
    model : modèle sklearn
        Modèle entraîné
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Valeurs réelles

    Retour
    ------
    metrics : dict
        Dictionnaire contenant les métriques
    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": mse,
        "r2": r2
    }

    return metrics
