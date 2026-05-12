# Superstore App

Application Flask unique pour le projet ML Global Superstore.

## Contenu

- DSO1 : segmentation clients avec K-Means.
- DSO2 : prédiction des ventes avec Random Forest.
- DSO3 : optimisation des ventes avec XGBoost.
- DSO4 : recommandations produits alignées avec la logique SVD du notebook.
- DSO5 : classification faible/forte des commandes.
- DSO6 : anomalies et niveaux d'anomalie avec Naive Bayes.

## Lancement

```powershell
cd ML\superstore-app
python app.py
```

URL locale : http://127.0.0.1:5000/

## Déploiement Render

Paramètres recommandés :

```text
Root Directory: ML/superstore-app
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

Variables d'environnement :

```text
SECRET_KEY=une-cle-secrete
```

Le dossier parent `ML` doit rester dans le dépôt GitHub, car l'application charge le dataset `Global_Superstore2.csv` et les modèles `.pkl` depuis ce dossier.
