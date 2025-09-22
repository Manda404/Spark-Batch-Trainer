Binary Classification Examples
==============================

Cette page montre comment utiliser **Spark Batch Trainer**
pour un problème de classification **binaire**.

---

XGBoost Example (Binary Classification)
---------------------------------------

.. code-block:: python

    from spark_batch_trainer.trainers.xgboost_trainer import XGBoostTrainer

    # 1. Instanciation
    trainer = XGBoostTrainer()
    target_column = "TARGET"

    # 2. Configuration du modèle
    config_model = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 6,
        "reg_lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 50,
    }

    # 3. Scheduler LR
    config_lr_scheduler = {
        "initial_lr": 0.1,
        "decay_rate": 0.25,
        "min_lr": 1e-3,
    }

    # 4. Entraînement batch-wise
    config_training = {
        "num_batches": 10,
        "max_patience": 2,
        "show_learning_curve": True,
        "use_sample_weight": True,
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

CatBoost Example (Binary Classification)
----------------------------------------

.. code-block:: python

    from spark_batch_trainer.trainers.catboost_trainer import CatBoostTrainer

    # 1. Instanciation
    trainer = CatBoostTrainer()
    target_column = "TARGET"

    # 2. Configuration du modèle
    config_model = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 100,
        "learning_rate": 0.01,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "random_seed": 42,
        "verbose": False,
    }

    # 3. Entraînement batch-wise
    config_training = {
        "num_batches": 15,
        "max_patience": 3,
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

LightGBM Example (Binary Classification)
----------------------------------------

.. code-block:: python

    from spark_batch_trainer.trainers.lightgbm_trainer import LGBMTrainer

    # 1. Instanciation
    trainer = LGBMTrainer()
    target_column = "TARGET"

    # 2. Configuration du modèle
    config_model = {
        "objective": "binary",
        "n_estimators": 50,
        "num_leaves": 30,
        "random_state": 42,
        "force_col_wise": True,
    }

    # 3. Scheduler LR
    config_lr_scheduler = {
        "initial_lr": 0.1,
        "decay_rate": 0.25,
        "min_lr": 1e-3,
    }

    # 4. Entraînement batch-wise
    config_training = {
        "num_batches": 5,
        "max_patience": 2,
        "eval_metric": "binary_logloss",
        "show_learning_curve": True,
        "use_sample_weight": True,
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
