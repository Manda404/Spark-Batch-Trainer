Présentation des datasets
=========================

Cette section présente les deux jeux de données utilisés dans les exemples
de **Spark Batch Trainer** :  

- Le **Diabetes Dataset** pour les tâches de **classification binaire**  
- Le **Obesity Dataset** pour les tâches de **classification multiclasse**  

.. note::

   Ces datasets sont utilisés uniquement à titre illustratif afin de montrer le
   fonctionnement du framework.  

---

1. Diabetes Dataset (Binary Classification)
-------------------------------------------

📌 **Objectif** : prédire si un patient est atteint de **diabète** en fonction
de caractéristiques médicales et démographiques.  

- **Source** : *National Institute of Diabetes and Digestive and Kidney Diseases* (Diabetes.csv) 
- **Taille** : `shape = (100000, 9)`  
- **Colonnes disponibles** :  

  - `gender` → Sexe du patient  
  - `age` → Âge  
  - `hypertension` → Antécédent d’hypertension (0 = Non, 1 = Oui)  
  - `heart_disease` → Antécédent de maladie cardiaque (0 = Non, 1 = Oui)  
  - `smoking_history` → Habitudes tabagiques  
  - `bmi` → Indice de masse corporelle  
  - `HbA1c_level` → Taux moyen de glycémie sur 3 mois  
  - `blood_glucose_level` → Niveau de glucose sanguin  
  - `diabetes` → **Variable cible (binaire : 0 = Non diabétique, 1 = Diabétique)**  

.. note::

   Ce dataset est utilisé pour illustrer la **classification binaire**
   dans les exemples d’entraînement avec **XGBoost**, **CatBoost** et
   **LightGBM**.

---

2. Obesity Dataset (Multiclass Classification)
----------------------------------------------

📌 **Objectif** : prédire la **catégorie de poids corporel** d’un individu
en fonction de ses habitudes alimentaires, son mode de vie et ses mesures
anthropométriques.  

- **Source** : Jeu de données académique sur l’obésité (ObesityDataset.csv)  
- **Taille** : `shape = (2111, 17)`  
- **Colonnes disponibles** :  

  - `Age` → Âge  
  - `Gender` → Sexe  
  - `Height` → Taille (mètres)  
  - `Weight` → Poids (kg)  
  - `CALC` → Consommation d’alcool  
  - `FAVC` → Consommation fréquente d’aliments riches en calories  
  - `FCVC` → Consommation de légumes  
  - `NCP` → Nombre de repas principaux par jour  
  - `SCC` → Surveillance des calories consommées  
  - `SMOKE` → Habitude tabagique  
  - `CH2O` → Consommation d’eau quotidienne  
  - `family_history_with_overweight` → Antécédents familiaux de surpoids  
  - `FAF` → Activité physique hebdomadaire  
  - `TUE` → Temps consacré à l’usage d’appareils électroniques  
  - `CAEC` → Grignotage entre les repas  
  - `MTRANS` → Mode de transport principal  
  - `NObeyesdad` → **Variable cible (multiclasse, 7 catégories)** :  

    - `Insufficient_Weight`  
    - `Normal_Weight`  
    - `Overweight_Level_I`  
    - `Overweight_Level_II`  
    - `Obesity_Type_I`  
    - `Obesity_Type_II`  
    - `Obesity_Type_III`  

.. note::

   Ce dataset est utilisé pour illustrer la **classification multiclasse**
   dans les exemples d’entraînement avec **XGBoost**, **CatBoost** et
   **LightGBM**.

---

3. Préparation et utilisation
-----------------------------

Avant d’utiliser **Spark Batch Trainer**, il est recommandé de suivre les étapes suivantes :  


1. **Charger les données**  
   Par exemple, avec `pandas.read_csv("fichier.csv")`.  

2. **Prétraiter les données (si nécessaire)** 

    - Encodage des variables catégorielles 

    - Gestion des valeurs manquantes  

    - Normalisation / standardisation des variables numériques  

3. **Diviser le dataset en sous-ensembles** 

    - **data_train** (60%) : sert à **entraîner le modèle** 

    - **data_valid** (20%) : sert à **valider les performances** et appliquer l’**early stopping**  

    - **data_test** (20%) : sert à l’**évaluation finale** sur des données jamais vues  

4. **Convertir les sous-ensembles train / validation** en **Spark DataFrames**
     
     Ces DataFrames seront directement utilisés par les méthodes `.fit()` du framework.  

---

📘 Ces deux datasets constituent la base des tutoriels :

   - :doc:`examples_binary` pour la **classification binaire** 
   - :doc:`examples_multiclass` pour la **classification multiclasse** 

