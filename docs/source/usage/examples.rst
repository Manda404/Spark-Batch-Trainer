Examples
========

Cette section illustre l’utilisation de **Spark Batch Trainer** avec plusieurs
frameworks de boosting en machine learning (**XGBoost**, **CatBoost**, **LightGBM**).

Deux cas de classification sont actuellement supportés :

- :doc:`examples_binary` → Exemple complet pour la **classification binaire**  
- :doc:`examples_multiclass` → Exemple complet pour la **classification multiclasse**

Chaque exemple présente de bout en bout :

- **Configuration du modèle** (*config_model* : hyperparamètres propres au framework)
- **Configuration de l’entraînement batch-wise** (*config_training* : nombre de batches, taille, patience, etc...)    
- **Configuration Scheduler du learning rate** (*config_lr_scheduler* : planification dynamique basée sur la **décroissance exponentielle**) 
- **Récupération modèle final** pour effectuer des predictions 

.. note::

   La version actuelle (**v1.0.0**) de **Spark Batch Trainer** est dédiée
   exclusivement aux tâches de **classification** (binaire et multiclasse),
   sur des jeux de données aussi bien **équilibrés** que **déséquilibrés**.  

   Le support d’autres tâches, telles que la **régression** ou le
   **ranking**, fait partie des évolutions envisagées dans les prochaines versions.