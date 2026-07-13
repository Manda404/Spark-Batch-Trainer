Binary classification
=====================

The following recipes assume two Spark DataFrames named ``train_df`` and
``validation_df``. Both contain the same feature columns and a binary target
named ``label``.

XGBoost
-------

.. code-block:: python

   from spark_batch_trainer import create_trainer

   trainer = create_trainer("xgboost")
   trainer.fit(
       train_dataframe=train_df,
       valid_dataframe=validation_df,
       target_column="label",
       model_config={
           "objective": "binary:logistic",
           "eval_metric": "logloss",
           "n_estimators": 100,
           "learning_rate": 0.05,
           "max_depth": 6,
           "random_state": 42,
       },
       training_config={
           "num_batches": 5,
           "max_patience": 3,
           "min_delta": 1e-4,
       },
   )

CatBoost
--------

.. code-block:: python

   trainer = create_trainer("catboost")
   trainer.fit(
       train_dataframe=train_df,
       valid_dataframe=validation_df,
       target_column="label",
       model_config={
           "loss_function": "Logloss",
           "eval_metric": "Logloss",
           "iterations": 100,
           "learning_rate": 0.05,
           "depth": 6,
           "random_seed": 42,
           "verbose": False,
       },
       training_config={"num_batches": 5, "max_patience": 3},
   )

LightGBM
--------

.. code-block:: python

   trainer = create_trainer("lightgbm")
   trainer.fit(
       train_dataframe=train_df,
       valid_dataframe=validation_df,
       target_column="label",
       model_config={
           "objective": "binary",
           "metric": "binary_logloss",
           "n_estimators": 100,
           "learning_rate": 0.05,
           "num_leaves": 31,
           "random_state": 42,
       },
       training_config={"num_batches": 5, "max_patience": 3},
   )

Inspect the result
------------------

``fit`` stores the selected model in the trainer. The history records the
batch-level metric sequences used for monitoring.

.. code-block:: python

   model = trainer.get_trained_model()
   history = trainer.get_training_history()

   print(history.batch_numbers)
   print(history.validation_scores)

.. important::

   Validation data and each training batch are collected in driver memory.
   Size partitions for the available driver capacity before starting a run.
