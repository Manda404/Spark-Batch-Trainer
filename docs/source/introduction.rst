Introduction
============

**Spark Batch Trainer** est un framework modulaire et extensible dédié à
l’entraînement par lots (**batch-wise training**) de modèles de boosting en machine learning,
tout en optimisant la gestion de la mémoire.

Il est conçu pour fonctionner avec :

- **Apache Spark DataFrames** : prétraitement distribué et scalable.  
  (les données d’entrée sont directement fournies sous forme de Spark DataFrames)  

- **Frameworks ML populaires** : intégration native avec **XGBoost**, **CatBoost** et **LightGBM**,  
  spécifiquement orientée vers les modèles de boosting.  

- **Scénarios Big Data** : apprentissage incrémental, équilibrage des classes,  
  suivi des **courbes d’apprentissage**.  

- **Cas d’usage ciblé** : cette première version est principalement conçue  
  pour adresser des problèmes de **classification**.  

Principales fonctionnalités
---------------------------

- Création de batchs stratifiés pour gérer efficacement les jeux de données déséquilibrés  
  et garantir la présence de toutes les classes dans chaque batch.  

- Prétraitement basique des données d’entraînement et de validation :  
  conversion fluide entre **Spark DataFrames** et **pandas DataFrames**,  
  gestion des conversions de types (par ex. *object* → *category*).  

- Visualisation des courbes d’apprentissage et des planifications du *learning rate*.  

👉 **Objectif** : offrir aux data scientists un outil simple, reproductible et extensible  
pour entraîner des modèles de boosting à grande échelle.
