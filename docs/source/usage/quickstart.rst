Quickstart
==========

This example assumes that ``train_df`` and ``validation_df`` are Spark
DataFrames with the same feature columns and a binary ``churn`` target.

.. code-block:: python

   from spark_batch_trainer import create_trainer

   trainer = create_trainer("xgboost")
   trainer.fit(
       train_dataframe=train_df,
       valid_dataframe=validation_df,
       target_column="churn",
       model_config={
           "objective": "binary:logistic",
           "eval_metric": "logloss",
           "n_estimators": 50,
           "learning_rate": 0.05,
           "max_depth": 6,
           "random_state": 42,
       },
       training_config={
           "num_batches": 5,
           "max_patience": 3,
           "metric_mode": "auto",
           "min_delta": 1e-4,
           "show_learning_curve": False,
       },
   )

   model = trainer.get_trained_model()
   history = trainer.get_training_history()

   print(history.batch_numbers)
   predictions = model.predict(validation_df.limit(100).toPandas().drop(columns=["churn"]))

.. warning::

   The final prediction snippet is intentionally small. Calling ``toPandas()``
   on an unbounded Spark DataFrame can exhaust driver memory.

Choose another backend
----------------------

Only the backend name and ``model_config`` need to change:

.. code-block:: python

   catboost_trainer = create_trainer("catboost")
   lightgbm_trainer = create_trainer("lightgbm")

Continue with :doc:`../configuration` and :doc:`../concepts` before running on
large datasets.
