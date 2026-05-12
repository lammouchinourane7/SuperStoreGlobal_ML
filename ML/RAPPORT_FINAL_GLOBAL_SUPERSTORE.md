# Rapport final — Projet Machine Learning Global Superstore

## Introduction

Ce rapport présente le travail réalisé dans le cadre du projet Machine Learning appliqué au dataset **Global Superstore**. Le projet vise à exploiter des données transactionnelles d’un supermarché mondial afin d’améliorer la prise de décision commerciale à travers plusieurs objectifs Data Science.

Le travail a été organisé selon la méthodologie **CRISP-DM**, qui structure un projet data en plusieurs phases : compréhension métier, compréhension des données, préparation des données, modélisation, évaluation et déploiement.

Le dataset utilisé est :

- **Nom du fichier :** `Global_Superstore2.csv`
- **Nombre de lignes initiales :** 51 290
- **Nombre de colonnes initiales :** 24
- **Domaine :** ventes, clients, produits, régions, profits, remises et livraison

Le projet est divisé en **6 DSO** :

| DSO | Objectif | Modèles principaux | Responsable |
|-----|----------|--------------------|-------------|
| DSO1 | Segmentation clients | K-Means, DBSCAN | Syrine Jedidi |
| DSO2 | Prédiction des ventes | Decision Tree, Random Forest | Alaa Tahri |
| DSO3 | Optimisation des ventes | AdaBoost, XGBoost | Aziz Msekni |
| DSO4 | Recommandation produits | SVD | Nourane Lammouchi |
| DSO5 | Classification des commandes | KNN, SVM | Wala Rezgui |
| DSO6 | Détection d’anomalies et régression | Isolation Forest, Naive Bayes, régressions | Ibtihel Mechergui |

L’objectif final est de construire un notebook complet et une application web permettant de présenter et tester les résultats des différents modèles.

## Compréhension métier

Le contexte métier concerne un supermarché mondial qui souhaite exploiter ses données transactionnelles pour améliorer ses performances commerciales.

Les données disponibles permettent d’analyser :

- les ventes par produit, région, catégorie et segment client ;
- la rentabilité des commandes ;
- les remises appliquées ;
- les coûts et délais de livraison ;
- les comportements d’achat des clients ;
- les commandes atypiques ou risquées.

La problématique principale est :

**Comment exploiter les données transactionnelles du dataset Global Superstore pour optimiser les performances commerciales d’un supermarché mondial grâce au Machine Learning ?**

Cette problématique est décomposée en **BOs** (*Business Objectives*) et **DSOs** (*Data Science Objectives*). Les BOs expriment les besoins métier, tandis que les DSOs traduisent ces besoins en objectifs techniques mesurables.

| BO — Business Objective | DSO — Data Science Objective | Modèles / méthodes | Résultat attendu |
|-------------------------|------------------------------|--------------------|------------------|
| Mieux comprendre les clients | DSO1 — Segmenter les clients | K-Means, DBSCAN | Identifier des profils clients exploitables |
| Anticiper les ventes | DSO2 — Prédire les ventes | Decision Tree, Random Forest | Estimer `Sales` avec une bonne précision |
| Améliorer la performance prédictive | DSO3 — Optimiser la prédiction des ventes | AdaBoost, XGBoost | Sélectionner le meilleur modèle de régression |
| Améliorer le cross-selling | DSO4 — Recommander des produits | SVD | Proposer des produits pertinents |
| Prioriser les commandes | DSO5 — Classifier les commandes | KNN, SVM | Distinguer commandes faibles et fortes |
| Surveiller les comportements atypiques | DSO6 — Détecter anomalies et délais | Isolation Forest, Naive Bayes, régressions | Identifier anomalies et analyser `Delivery_Days` |

Une autre lecture synthétique des besoins est la suivante :

| Besoin métier | Objectif Data Science |
|--------------|------------------------|
| Comprendre les profils clients | Segmenter les clients selon leurs comportements d’achat |
| Anticiper les ventes | Prédire le montant des ventes d’une commande |
| Améliorer la précision des prédictions | Tester des modèles avancés comme XGBoost |
| Améliorer le cross-selling | Recommander des produits pertinents |
| Identifier les commandes importantes | Classifier les commandes faibles ou fortes |
| Surveiller les comportements atypiques | Détecter les anomalies et analyser les délais |

Ainsi, chaque DSO répond à un besoin métier spécifique tout en utilisant le même dataset.

## Compréhension des données

Le dataset **Global Superstore** contient des informations relatives aux commandes passées dans différentes régions du monde.

Les principales colonnes sont :

| Colonne | Description |
|--------|-------------|
| `Order Date` | Date de commande |
| `Ship Date` | Date d’expédition |
| `Ship Mode` | Mode d’expédition |
| `Customer ID` | Identifiant client |
| `Customer Name` | Nom du client |
| `Segment` | Segment client |
| `City`, `State`, `Country` | Localisation |
| `Market`, `Region` | Marché et région |
| `Category`, `Sub-Category` | Catégorie et sous-catégorie produit |
| `Product Name` | Nom du produit |
| `Sales` | Montant des ventes |
| `Quantity` | Quantité commandée |
| `Discount` | Remise appliquée |
| `Profit` | Profit réalisé |
| `Shipping Cost` | Coût de livraison |
| `Order Priority` | Priorité de la commande |

L’analyse exploratoire a permis d’identifier :

- la taille du dataset : **51 290 lignes** ;
- la présence de colonnes numériques et catégorielles ;
- une colonne `Postal Code` fortement incomplète ;
- des variables importantes pour la modélisation : `Sales`, `Profit`, `Quantity`, `Discount`, `Shipping Cost` ;
- des variables catégorielles utiles : `Region`, `Category`, `Sub-Category`, `Segment`, `Ship Mode`.

Des visualisations ont été réalisées :

- histogrammes des variables numériques ;
- boxplots pour détecter les outliers ;
- countplots pour les variables catégorielles ;
- matrice de corrélation ;
- analyse par segment, région et catégorie ;
- ACP pour comprendre les relations entre variables quantitatives.

L’analyse a montré que les ventes et profits sont influencés par plusieurs facteurs : quantité, coût de livraison, remise, catégorie produit, région et comportement client.

## Préparation des données

La phase de préparation a permis de nettoyer et transformer les données afin de les rendre exploitables pour les modèles.

Les principales étapes réalisées sont :

1. Conversion des dates :

   - `Order Date` et `Ship Date` ont été converties en format date.
   - Une nouvelle variable `Delivery_Days` a été créée :

   ```text
   Delivery_Days = Ship Date - Order Date
   ```

2. Suppression des colonnes peu utiles ou non adaptées :

   - `Row ID`
   - `Order ID`
   - `Customer Name`
   - `Product ID`
   - `Postal Code`

3. Filtrage métier :

   - conservation des quantités positives ;
   - conservation des ventes positives ;
   - contrôle des remises ;
   - contrôle du coût de livraison par rapport aux ventes.

4. Gestion des outliers :

   Les outliers ont été traités avec la méthode IQR sur les colonnes :

   - `Sales`
   - `Profit`
   - `Discount`
   - `Shipping Cost`

5. Création de variables dérivées :

   - `Delivery_Days`
   - `Profit_Margin`
   - `Discount_Rate`
   - `Order_Year`
   - `Order_Month`
   - `Order_Day`
   - `Order_Week`

6. Encodage des variables catégorielles :

   Les variables catégorielles ont été transformées afin d’être utilisables par les modèles :

   - encodage des colonnes texte ;
   - création de variables binaires pour certaines catégories ;
   - standardisation ou normalisation selon les modèles.

Après préparation et suppression des outliers, le nombre de lignes utilisées pour la modélisation est :

```text
14 618 observations
```

## Modélisation

La phase de modélisation regroupe les 6 DSO du projet.

### DSO1 — Segmentation clients

Objectif : regrouper les clients en segments homogènes selon leur comportement d’achat.

Les données ont été agrégées par client avec les variables suivantes :

- `Total_Sales`
- `Avg_Order_Value`
- `Order_Count`
- `Total_Profit`
- `Avg_Profit_Margin`
- `Avg_Discount_Rate`
- `Avg_Delivery_Days`

Modèles utilisés :

- **K-Means**, avec `k = 4`
- **DBSCAN**

Résultats :

| Modèle | Métrique | Résultat |
|--------|----------|----------|
| K-Means | Silhouette | 0.2770 |
| K-Means | Davies-Bouldin | 1.3209 |
| DBSCAN | Silhouette | 0.3404 |
| DBSCAN | Davies-Bouldin | 2.7373 |
| DBSCAN | Bruit détecté | 10.2 % |

Interprétation :

- DBSCAN donne une meilleure silhouette.
- K-Means donne un meilleur score Davies-Bouldin, ce qui indique des clusters plus compacts et mieux séparés selon cette métrique.
- DBSCAN détecte automatiquement du bruit, mais il isole **10.2 %** des clients comme points bruités, ce qui complique l’interprétation métier et l’utilisation opérationnelle.
- K-Means produit exactement **4 segments**, ce qui correspond au besoin métier : créer des groupes lisibles, stables et activables par les équipes marketing.
- K-Means a donc été retenu pour l’application web, non pas parce qu’il domine toutes les métriques, mais parce qu’il offre le meilleur compromis entre performance, stabilité et interprétabilité métier.

### DSO2 — Prédiction des ventes

Objectif : prédire le montant des ventes `Sales`.

Modèles utilisés :

- **Decision Tree Regressor**
- **Random Forest Regressor**

Résultats :

| Modèle | MAE | RMSE | R² |
|--------|-----|------|----|
| Decision Tree | 29.1992 | 54.9796 | 0.4856 |
| Random Forest | 20.3980 | 37.6652 | 0.7586 |

Interprétation :

Random Forest donne de meilleurs résultats que l’arbre de décision simple. Il réduit l’overfitting et améliore la précision des prédictions.

### DSO3 — Optimisation des ventes

Objectif : améliorer la prédiction des ventes avec des modèles de boosting.

Modèles utilisés :

- **AdaBoost Regressor**
- **XGBoost Regressor**

Résultats :

| Modèle | MAE | RMSE | R² |
|--------|-----|------|----|
| AdaBoost | 31.1266 | 48.4058 | 0.6013 |
| XGBoost | 18.4326 | 34.1136 | 0.8020 |

Interprétation :

XGBoost est le meilleur modèle de régression du projet. Il obtient le meilleur R² et le RMSE le plus faible pour la prédiction des ventes.

Le choix de XGBoost est donc défendable avec les deux métriques : **le RMSE diminue** par rapport à AdaBoost, ce qui signifie une erreur moyenne plus faible en valeur de ventes, et **le R² augmente**, ce qui signifie que le modèle explique mieux la variance de `Sales`. Dans la synthèse globale, le RMSE est mis en avant car il est directement interprétable comme une erreur de prédiction sur la cible métier.

### DSO4 — Recommandation produits

Objectif : recommander des produits à partir de la matrice client-produit.

Modèle utilisé :

- **SVD**, via `TruncatedSVD`

Le système construit une matrice client-produit basée sur les interactions d’achat, puis utilise une réduction de dimension pour estimer des recommandations.

Résultats :

| Indicateur | Résultat |
|-----------|----------|
| RMSE SVD | 5.32 |
| Precision@5 | 0.0011 |

Interprétation :

La Precision@5 est faible, ce qui montre que les recommandations peuvent être améliorées avec davantage de données ou de signaux supplémentaires, comme les avis clients, les historiques de navigation ou la saisonnalité.

Pour ce DSO, aucune alternative de recommandation n’a été testée dans le notebook. **SVD est donc retenu comme baseline**, c’est-à-dire comme première approche de référence pour construire une matrice client-produit et générer des recommandations.

### DSO5 — Classification des commandes

Objectif : classifier les commandes en classes faibles ou fortes selon le montant des ventes.

Modèles utilisés :

- **KNN**
- **SVM**

Résultats :

| Modèle | Accuracy |
|--------|----------|
| KNN | 0.8157 |
| SVM | 0.9056 |

Interprétation :

SVM donne la meilleure performance avec une accuracy de **0.9056**. Il est donc plus adapté que KNN pour séparer les commandes faibles et fortes.

### DSO6 — Détection d’anomalies et régression

Objectif : détecter les commandes atypiques et analyser les délais de livraison.

Modèles utilisés :

- **Isolation Forest**
- **Gaussian Naive Bayes**
- **Régression linéaire**
- **Régression polynomiale**

Résultats :

| Approche | Métrique | Résultat |
|----------|----------|----------|
| Isolation Forest | Anomalies détectées | 293 observations |
| Isolation Forest | Taux d’anomalies | 2.00 % |
| Naive Bayes | Accuracy | 0.62 |
| Naive Bayes | AUC macro | 0.6246 |
| Régression linéaire | MAE | 1.1587 |
| Régression linéaire | RMSE | 1.4765 |
| Régression linéaire | R² | 0.2258 |
| Régression polynomiale | MAE | 1.1147 |
| Régression polynomiale | RMSE | 1.4182 |
| Régression polynomiale | R² | 0.2858 |

Interprétation :

Isolation Forest permet d’identifier une proportion limitée de commandes atypiques. Naive Bayes donne une classification moyenne des niveaux d’anomalie. La régression polynomiale améliore légèrement la prédiction des délais par rapport à la régression linéaire.

### Synthèse globale des modèles

| DSO | Modèle retenu | Métrique principale | Résultat |
|-----|---------------|---------------------|----------|
| DSO1 | K-Means | Silhouette | 0.2770 |
| DSO2 | Random Forest | R² | 0.7586 |
| DSO3 | XGBoost | RMSE | 34.1136 |
| DSO4 | SVD | Precision@5 | 0.0011 |
| DSO5 | SVM | Accuracy | 0.9056 |
| DSO6 | Isolation Forest + Naive Bayes | AUC macro | 0.6246 |

Le meilleur modèle pour la prédiction des ventes est **XGBoost**, avec :

- MAE = 18.4326
- RMSE = 34.1136
- R² = 0.8020

### Benchmarking final et justification des choix

Le benchmarking ne se limite pas à choisir le score le plus élevé. Les modèles ont aussi été comparés selon leur interprétabilité, leur stabilité, leur capacité à être intégrés dans l’application web et leur cohérence avec l’objectif métier.

| DSO | Modèles comparés | Modèle retenu | Justification |
|-----|------------------|---------------|---------------|
| DSO1 | K-Means vs DBSCAN | K-Means | Meilleur compromis entre stabilité, interprétabilité et segmentation exploitable en 4 groupes |
| DSO2 | Decision Tree vs Random Forest | Random Forest | Meilleur R² et RMSE plus faible que l’arbre seul |
| DSO3 | AdaBoost vs XGBoost | XGBoost | Meilleur modèle global pour prédire les ventes, R² = 0.8020 |
| DSO4 | Aucune alternative testée | SVD baseline | SVD est retenu comme première approche de référence pour la recommandation client-produit |
| DSO5 | KNN vs SVM | SVM | Accuracy nettement supérieure : 0.9056 |
| DSO6 | Isolation Forest, Naive Bayes, régressions | Combinaison des approches | Isolation Forest pour détecter les anomalies, Naive Bayes pour les niveaux d’anomalie, régressions pour `Delivery_Days` |

## Déploiement

Une application web Flask a été développée afin de présenter les résultats et rendre les modèles plus accessibles.

**Statut du déploiement :** l’application est actuellement déployée en **local** pour la démonstration technique, via `http://127.0.0.1:5000/`. Elle n’est pas encore publiée sur un serveur cloud accessible publiquement. Pour la soutenance, la démonstration se fait donc depuis la machine locale, avec l’environnement Python et les modèles exportés.

Le déploiement local reste pertinent pour valider :

- le chargement des modèles exportés ;
- l’intégration entre le front-end Flask et la partie IA ;
- la cohérence des formulaires avec le notebook ;
- l’affichage des résultats et métriques pour chaque DSO.

Un déploiement cloud pourra être réalisé ensuite sur Render, Railway, Azure, AWS, PythonAnywhere ou un serveur universitaire.

Le dossier principal de l’application est :

```text
ML/superstore-app
```

L’application s’appelle **Superstore App**.

Elle contient :

- une page d’accueil ;
- un dashboard global ;
- une page pour chaque DSO ;
- des formulaires de test ;
- des résultats numériques issus du notebook ;
- une interface homogène et adaptée au dataset Global Superstore.

Les pages principales sont :

| Page | Description |
|------|-------------|
| Accueil | Présentation générale du projet |
| Dashboard | Synthèse des DSO, modèles et métriques |
| DSO1 | Test de segmentation client |
| DSO2 | Prédiction des ventes avec Random Forest |
| DSO3 | Prédiction des ventes avec XGBoost |
| DSO4 | Recommandation produits |
| DSO5 | Classification faible/forte des commandes |
| DSO6 | Détection de niveau d’anomalie |

Lancement local :

```powershell
cd ML\superstore-app
python app.py
```

URL locale :

```text
http://127.0.0.1:5000/
```

Les modèles exportés depuis le notebook sont :

| Fichier | Utilisation |
|--------|-------------|
| `dso1_kmeans.pkl` | Segmentation clients |
| `dso1_scaler_cluster.pkl` | Standardisation DSO1 |
| `dso2_random_forest.pkl` | Prédiction des ventes |
| `dso3_xgboost.pkl` | Optimisation des ventes |
| `dso6_isolation_forest.pkl` | Détection d’anomalies |
| `dso6_naive_bayes.pkl` | Classification des niveaux d’anomalie |

L’application permet donc de transformer le notebook en une interface exploitable pour une démonstration métier.

## Conclusion et perspectives

Ce projet a permis de construire une solution Machine Learning complète sur le dataset **Global Superstore**.

Les principales réalisations sont :

- exploration et compréhension du dataset ;
- nettoyage et préparation des données ;
- création de variables utiles comme `Delivery_Days`, `Profit_Margin` et `Discount_Rate` ;
- modélisation de 6 objectifs Data Science ;
- comparaison des performances ;
- sélection des modèles les plus efficaces ;
- création d’une application web Flask pour la démonstration.

Les résultats montrent que :

- **K-Means** est exploitable pour segmenter les clients ;
- **Random Forest** donne une bonne prédiction des ventes ;
- **XGBoost** est le meilleur modèle de régression ;
- **SVD** permet de construire une première logique de recommandation ;
- **SVM** est performant pour classifier les commandes ;
- **Isolation Forest** détecte les commandes atypiques ;
- **Naive Bayes** permet une classification des niveaux d’anomalie.

### Perspectives d’amélioration

Plusieurs améliorations peuvent être envisagées :

1. Exporter des pipelines complets incluant preprocessing et modèles.
2. Améliorer le système de recommandation avec plus de signaux client.
3. Ajouter une interprétabilité des modèles avec SHAP ou permutation importance.
4. Mettre en place un suivi MLOps : monitoring, réentraînement et versioning.
5. Ajouter une base de données pour historiser les prédictions de l’application.
6. Déployer l’application sur une plateforme cloud.
7. Ajouter une authentification pour un usage professionnel.

En conclusion, le projet montre comment les données transactionnelles de Global Superstore peuvent être exploitées pour répondre à plusieurs problématiques commerciales importantes : segmentation, prévision, recommandation, classification et détection d’anomalies.
