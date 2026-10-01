from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, save_model
from src.evaluation import evaluate_model
from src.visualization import plot_real_vs_pred, plot_feature_importance


def main():
    # 1️⃣ Chargement des données
    data_path = "data/raw/Housing.csv"
    df = load_data(data_path)

    # 2️⃣ Préprocessing
    X, y = preprocess_data(df)

    # 3️⃣ Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Entraînement
    model = train_model(X_train, y_train)

    # 5️⃣ Évaluation
    metrics = evaluate_model(model, X_test, y_test)
    print("📊 Résultats du modèle")
    print(f"MSE : {metrics['mse']:.2f}")
    print(f"R²  : {metrics['r2']:.4f}")

    # 6️⃣ Visualisations
    y_pred = model.predict(X_test)
    plot_real_vs_pred(y_test, y_pred)
    plot_feature_importance(model, X_train.columns)

    # 7️⃣ Sauvegarde du modèle
    model_path = "models/linear_regression.pkl"
    save_model(model, model_path)
    print(f"✅ Modèle sauvegardé dans {model_path}")


if __name__ == "__main__":
    main()
