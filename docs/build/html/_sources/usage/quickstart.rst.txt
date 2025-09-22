Quickstart
==========

Cet exemple minimal montre comment utiliser **Spark Batch Trainer** avec XGBoost
sur un problème de classification binaire.

.. code-block:: python

   from spark_batch_trainer.trainers.xgboost_trainer import XGBoostTrainer

   trainer = XGBoostTrainer()

   config_model = {
       "objective": "binary:logistic",
       "eval_metric": "logloss",
       "n_estimators": 50,
   }

   config_training = {
       "num_batches": 5,
       "show_learning_curve": True,
   }

   trainer.fit(
       train_dataframe=spark_train_df,
       valid_dataframe=spark_valid_df,
       target_column="TARGET",
       config_training=config_training,
       config_model=config_model,
   )

   final_model = trainer.get_trained_model()