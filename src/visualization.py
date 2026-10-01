import matplotlib.pyplot as plt
import seaborn as sns


def plot_real_vs_pred(y_test, y_pred):
    """
    Graphique des valeurs réelles vs prédites.

    Paramètres
    ----------
    y_test : pd.Series ou array-like
        Valeurs réelles
    y_pred : pd.Series ou array-like
        Valeurs prédites
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Réel vs Prédit")

    # Ligne parfaite y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Affiche l'importance des variables (coefficients).

    Paramètres
    ----------
    model : LinearRegression
        Modèle entraîné
    feature_names : list
        Noms des features
    """

    importance = model.coef_

    sns.barplot(x=importance, y=feature_names)
    plt.title("Importance des variables")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")

    plt.show()
