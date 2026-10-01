import pandas as pd
import os


def load_data(path: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier CSV.

    Paramètres
    ----------
    csv_path : str
        Chemin vers le fichier CSV (ex: data/raw/Housing.csv)

    Retour
    ------
    pd.DataFrame
        Dataset chargé sous forme de DataFrame pandas
    """

    # Vérifier que le fichier existe
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Le fichier {path} est introuvable. Vérifie le chemin."
        )

    # Charger le CSV
    df = pd.read_csv(path)

    # Vérification minimale
    if df.empty:
        raise ValueError("Le fichier CSV est vide.")

    return df
