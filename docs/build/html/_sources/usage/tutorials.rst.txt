Tutoriels
=========

Cette section présente les concepts fondamentaux de **Spark Batch Trainer**,
indispensables pour comprendre son architecture et son mode d’utilisation.  

📘 Les cas pratiques complets et reproductibles (datasets, code exécutable)
sont détaillés séparément dans la section :doc:`examples`.

---

1. Flux de travail général
--------------------------

L’utilisation de **Spark Batch Trainer** repose sur un flux de travail standardisé :  

1. Préparer les données d’entrée au format **Spark DataFrame** (train et validation)  
2. Définir les dictionnaires de configuration  
3. Instancier un **trainer** adapté au framework ML choisi (**XGBoost**, **CatBoost**, **LightGBM**)  
4. Lancer l’entraînement batch-wise et évaluer le modèle obtenu  

.. note::

   Le découpage en *train/validation/test* n’est pas géré directement par
   Spark Batch Trainer.  
   L’utilisateur doit fournir au minimum un **train** et un **validation set**
   sous forme de **Spark DataFrames** déjà prétraités.

---

2. Préparation des données
--------------------------

Le framework attend en entrée des **Spark DataFrames déjà prétraités**.  

.. note::

   Spark Batch Trainer ne réalise pas de **prétraitement complet**
   (nettoyage, encodage, normalisation, gestion des valeurs manquantes, etc.).  
   Ces étapes doivent être effectuées en amont par l’utilisateur.

En revanche, le framework offre certaines fonctionnalités automatiques :  

- Conversion des colonnes de type *object* en **catégoriel**  
- Détection des colonnes catégorielles et ajustement des hyperparamètres
  associés dans le modèle sous-jacent (par ex. gestion native dans XGBoost,
  CatBoost ou LightGBM)  

Ainsi, l’utilisateur conserve la maîtrise du prétraitement tout en bénéficiant
d’une intégration simplifiée avec les frameworks ML supportés.

---

3. Les dictionnaires de configuration
-------------------------------------

L’entraînement est piloté par trois dictionnaires distincts.  
Chaque dictionnaire joue un rôle précis et peut être laissé vide (`{}`) pour
utiliser les paramètres par défaut prévus dans le framework.

--- a) `config_model`

Ce dictionnaire définit les **hyperparamètres du modèle ML** choisi
(**XGBoost**, **CatBoost**, **LightGBM**).  

📌 Pourquoi le définir ?

- Adapter le modèle au type de tâche (binaire, multiclasse, régression)  
- Optimiser la performance (profondeur des arbres, régularisation, taux
  d’échantillonnage, etc.)  
- Garantir la reproductibilité grâce au `random_state`  

.. warning::

   En classification **multiclasse**, il est fortement recommandé de définir
   explicitement ce dictionnaire afin d’éviter des erreurs ou une
   dégradation de performance lors de l’entraînement.

--- b) `config_training`

Ce dictionnaire contrôle les paramètres liés à l’**entraînement batch-wise**.  

📌 Pourquoi le définir ?

- Spécifier le nombre de lots (`num_batches`) pour gérer de grands datasets  
- Définir une patience globale (`max_patience`) pour l’early stopping  
- Activer le suivi des courbes d’apprentissage (`show_learning_curve`)
  afin de visualiser l’évolution de la performance en train/validation  
- Gérer les classes déséquilibrées en activant (`use_sample_weight`)  

.. note::

   Si ce dictionnaire est laissé vide (`{}`), le framework applique ses
   paramètres internes par défaut.

--- c) `config_lr_scheduler`

Ce dictionnaire définit la stratégie de **planification dynamique du learning rate**.  

📌 Qu’est-ce qu’un *learning rate scheduler* ?  

Un *learning rate scheduler* est une méthode qui ajuste le **taux d’apprentissage**
au cours de l’entraînement.

Il permet de commencer avec un taux élevé pour effectuer des mises à jour rapides
quand les paramètres sont encore loin de leur optimum, puis de réduire ce taux
au fur et à mesure que le modèle s’approche de la solution optimale afin
d’affiner l’entraînement.

📌 Pourquoi l’utiliser ?  

- Démarrer avec un taux élevé pour accélérer la convergence 

- Réduire progressivement ce taux pour stabiliser le modèle  

- Limiter le surapprentissage sur des données complexes ou bruitées  

Exemple de stratégie : **décroissance exponentielle**, avec :  

- `initial_lr` → taux d’apprentissage de départ  
- `decay_rate` → facteur de réduction appliqué à chaque étape  
- `min_lr` → valeur minimale en dessous de laquelle le taux ne descend pas  

Exemple alternatif : **Step Decay**  
Réduction par paliers, où le learning rate est divisé par un facteur fixe toutes
les *s* itérations.  

.. note::

   Dans cette version de **Spark Batch Trainer**, seule la stratégie de
   **décroissance exponentielle** est implémentée.

   Si ce dictionnaire est laissé vide (`{}`), le modèle conserve son
   learning rate fixe par défaut.

---

4. Les trainers
---------------

**Spark Batch Trainer** fournit plusieurs implémentations de trainers, chacune
associée à un framework de boosting :  

- `XGBoostTrainer` (**XGBoost**) :  
  Performant et flexible, particulièrement adapté aux datasets hétérogènes et
  aux tâches de classification binaire.  

- `CatBoostTrainer` (**CatBoost**) :  
  Spécialement optimisé pour la gestion native des **variables catégorielles**,
  réduisant le besoin de prétraitement manuel.  

- `LightGBMTrainer` (**LightGBM**) :  
  Conçu pour la **vitesse et l’efficacité mémoire**, performant sur de très
  grands datasets avec un temps d’entraînement réduit.  

Tous héritent d’une classe abstraite commune (`BaseTrainer`) et partagent une
interface unifiée :  

- `.fit()` → lance l’entraînement  
- `.get_trained_model()` → récupère le modèle entraîné  

Cette conception modulaire garantit une API homogène et permet d’ajouter
facilement de nouveaux trainers.

---

5. Entraînement batch-wise
--------------------------

Le cœur du framework repose sur l’entraînement par lots (**batch-wise
training**).  

Plutôt que d’entraîner un modèle sur l’ensemble des données en une seule fois,
le dataset est divisé en **batches successifs**.  
Chaque batch est utilisé pour affiner le modèle.  

📌 Avantages :  

- Réduction de la consommation mémoire  
- Meilleure gestion des datasets massifs  
- Possibilité d’appliquer un **early stopping global** (`max_patience`)  
- Suivi détaillé des courbes d’apprentissage au fil des batches  

Cette approche rend le framework adapté aux environnements **Big Data**
et aux contextes de production à grande échelle.

---

6. Évaluation et récupération du modèle
---------------------------------------

À la fin de l’entraînement, l’utilisateur peut :  

- Récupérer le modèle entraîné via `.get_trained_model()`  
- Effectuer des prédictions sur de nouvelles données  
- Visualiser les courbes d’apprentissage pour diagnostiquer les performances  
- Exporter ou sauvegarder le modèle en utilisant les méthodes natives du
  framework sous-jacent (ex. `.save_model()` pour XGBoost ou LightGBM)  

L’évaluation repose sur les données de test définies en amont, ce qui garantit
une mesure objective des performances.

---

7. Bonnes pratiques
-------------------

Pour tirer le meilleur parti de **Spark Batch Trainer**, il est recommandé de :  

- Vérifier systématiquement vos splits (train/valid/test) afin d’éviter toute
  fuite de données  

- Toujours définir `config_model` aussi bien pour la classification binaire que
  pour la classification multiclasse  

- Activer `show_learning_curve=True` dans `config_training` pour surveiller le
  surapprentissage et visualiser les courbes train vs validation  

- Exploiter les pondérations (`use_sample_weight`) en cas de datasets
  déséquilibrés  

- Sauvegarder vos modèles via les méthodes natives du framework ML sous-jacent  

---

📘 Pour des cas d’utilisation complets (datasets, code reproductible,
résultats), consultez la section :doc:`examples`.
