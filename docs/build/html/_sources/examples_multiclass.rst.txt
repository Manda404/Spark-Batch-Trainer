Multiclass Classification Examples
==================================

Cette page montre comment utiliser **Spark Batch Trainer**
pour un problème de classification **multiclasse**.

---

XGBoost Example (Multiclass Classification)
-------------------------------------------

.. code-block:: python

    from spark_batch_trainer.trainers.xgboost_trainer import XGBoostTrainer

    # 1. Instanciation
    trainer = XGBoostTrainer()
    target_column = "TARGET"

    # 2. Configuration du modèle
    config_model = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "reg_lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 10,
    }

    # 3. Scheduler LR
    config_lr_scheduler = {
        "initial_lr": 0.1,
        "decay_rate": 0.25,
        "min_lr": 1e-3,
    }

    # 4. Entraînement batch-wise
    config_training = {
        "num_batches": 2,
        "max_patience": 1,
        "show_learning_curve": True,
        "use_sample_weight": True,
        "verbose": 100,
    }

    # 5. Fit & Evaluation
    trainer.fit(
        train_dataframe=spark_train_df,
        valid_dataframe=spark_valid_df,
        target_column=target_column,
        config_training=config_training,
        config_model=config_model,
        config_lr_scheduler=config_lr_scheduler,
    )

    # 6. Récupération modèle entraîné
    final_model = trainer.get_trained_model()

---

CatBoost Example (Multiclass Classification)
--------------------------------------------

.. code-block:: python

    from spark_batch_trainer.trainers.catboost_trainer import CatBoostTrainer

    # 1. Instanciation
    trainer = CatBoostTrainer()
    target_column = "TARGET"

    # 2. Configuration du modèle
    config_model = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "random_seed": 42,
        "verbose": 100,
    }

    # 3. Entraînement batch-wise
    config_training = {
        "num_batches": 3,
        "max_patience": 2,
        "show_learning_curve": True,
    }

    # 4. Fit & Evaluation
    trainer.fit(
        train_dataframe=spark_train_df,
        valid_dataframe=spark_valid_df,
        target_column=target_column,
        config_training=config_training,
        config_model=config_model,
    )

    # 5. Récupération modèle entraîné
    final_model = trainer.get_trained_model()

---

LightGBM Example (Multiclass Classification)
--------------------------------------------

.. code-block:: python

    from spark_batch_trainer.trainers.lightgbm_trainer import LGBMTrainer

    # 1. Instanciation
    trainer = LGBMTrainer()
    target_column = "TARGET"

    # 2. Configuration du modèle
    config_model = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(encoder.classes_),
        "n_estimators": 500,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": -1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "force_col_wise": True,
        "verbose": -1,
    }

    # 3. Scheduler LR
    config_lr_scheduler = {
        "initial_lr": 0.1,
        "decay_rate": 0.25,
        "min_lr": 1e-3,
    }

    # 4. Entraînement batch-wise
    config_training = {
        "num_batches": 2,
        "max_patience": 1,
        "show_learning_curve": True,
        "use_sample_weight": True,
        "eval_metric": "multi_logloss",
    }

    # 5. Fit & Evaluation
    trainer.fit(
        train_dataframe=spark_train_df,
        valid_dataframe=spark_valid_df,
        target_column=target_column,
        config_training=config_training,
        config_model=config_model,
        config_lr_scheduler=config_lr_scheduler,
    )

    # 6. Récupération modèle entraîné
    final_model = trainer.get_trained_model()