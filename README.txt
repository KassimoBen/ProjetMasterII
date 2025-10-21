
DataLab Flask – Application web d'analyse de données (Python + Flask)

Fonctionnalités principales
- Upload CSV/Excel + jeux d'essai (Iris, Ventes)
- Aperçu + profil rapide (types, manquants)
- Nettoyage: lignes/colonnes vides, doublons, remplissage NA (fixe/moyenne/médiane/mode), filtres query
- Transformation: sélectionner colonnes, renommer, convertir types, colonne calculée (pandas.eval)
- Analyse: corrélation (heatmap Plotly), ANOVA (scipy), régression linéaire (scikit-learn)
- Pivot: tableau croisé avec agrégations
- Normalisation: MinMax/Standard/Robust (scikit-learn)
- Export: CSV, Excel, JSON
- Projet/Rapport: export .zip (data.csv + pipeline.json + meta.json)
- Undo/Redo/Reset via historique en mémoire

Installation et lancement
1) Dans un environnement Python 3.9+ :
   pip install -r requirements.txt
2) Lancer :
   python app.py
3) Ouvrir http://127.0.0.1:5000

Astuce production
- Déployer avec Gunicorn + reverse proxy, et stocker les données côté serveur (base/Redis) plutôt qu'en mémoire.
