Introduction
============

**Spark Batch Trainer** est un framework modulaire et extensible pour
l’entraînement de modèles de machine learning par lots (**batch-wise training**).

Il est conçu pour fonctionner avec :

- **Apache Spark DataFrames** : prétraitement distribué et scalable.
- **Frameworks ML populaires** : intégration avec **XGBoost**, **CatBoost**, et **LightGBM**.
- **Scénarios Big Data** : apprentissage incrémental, équilibrage des classes, suivi des métriques.

Principales fonctionnalités
---------------------------

- Création de batchs stratifiés pour gérer les datasets déséquilibrés.

- Conversion fluide entre **Spark DataFrames** et **pandas DataFrames**.

- Visualisation des courbes d’apprentissage et des planifications de learning rate.

- Extraction des métriques multi-frameworks.

👉 Objectif : permettre aux data scientists d’entraîner des modèles à grande échelle
de façon simple, reproductible et extensible.
