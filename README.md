# Prédiction du prix des maisons (régression linéaire)

Petit projet fait en ING2 pour pratiquer la régression linéaire sur un vrai jeu de données.

L'idée est simple : à partir de quelques infos sur une maison (taille du terrain, nombre de chambres, s'il y a un garage, la clim...), essayer de deviner son prix de vente.

## Les données

Le fichier `data/raw/Housing.csv` contient 546 maisons vendues à Windsor, au Canada. Pour chaque maison on a :

- `price` : le prix de vente (c'est ce qu'on cherche à prédire)
- `lotsize` : la surface du terrain
- `bedrooms`, `bathrms`, `stories` : chambres, salles de bain, étages
- `driveway`, `recroom`, `fullbase`, `gashw`, `airco`, `prefarea` : des colonnes oui/non (allée, salle de jeux, sous-sol, chauffe-eau au gaz, clim, quartier recherché)
- `garagepl` : le nombre de places de garage

## Comment c'est organisé

```
data/raw/       le CSV
notebooks/      l'exploration, les graphes et les premiers essais de modèle
src/            le code "propre", découpé en petits fichiers
models/         le modèle entraîné (.pkl)
main.py         lance tout d'un coup
```

J'ai commencé par les notebooks pour comprendre les données, puis j'ai remis le code au propre dans `src/` :

- `data_loader.py` lit le CSV
- `preprocessing.py` enlève la colonne d'index et transforme les yes/no en 1/0
- `model.py` entraîne la régression et la sauvegarde
- `evaluation.py` calcule le MSE et le R²
- `visualization.py` affiche les prix réels vs prédits et le poids de chaque variable

## Lancer le projet

```bash
pip install -r requirements.txt
python main.py
```

Ça charge les données, entraîne le modèle sur 80 % des maisons, le teste sur les 20 % restantes, affiche les résultats et deux graphes, puis enregistre le modèle dans `models/`.

## Résultats

Sur le jeu de test j'obtiens un R² d'environ 0.62, donc le modèle explique un peu plus de la moitié des variations de prix. L'erreur moyenne tourne autour de 16 000 $.

Ce n'est pas parfait, mais pour un modèle linéaire sans aucun réglage c'est plutôt correct. Pour aller plus loin on pourrait passer le prix en log, normaliser les variables ou tester d'autres modèles (Ridge, Random Forest...).
