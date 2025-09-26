Quickstart
==========

This minimal example shows how to train an **XGBoost** model with **Spark Batch Trainer**
on a binary classification problem.  
You must provide two Spark DataFrames: a **train set**, a **validation set**, and the target column.

.. code-block:: python

   from spark_batch_trainer.trainers.xgboost_trainer import XGBoostTrainer

   # Define model and training configurations
   config_model = {
       "objective": "binary:logistic",
       "eval_metric": "logloss",
       "n_estimators": 50,
   }

   config_training = {
       "num_batches": 5,
       "show_learning_curve": True,
   }

   # Instantiate the trainer
   trainer = XGBoostTrainer()

   # Fit the model on train and validation DataFrames
   trainer.fit(
       train_dataframe=spark_train_df,
       valid_dataframe=spark_valid_df,
       target_column="TARGET",
       config_training=config_training,
       config_model=config_model,
   )

   # Retrieve the trained model
   final_model = trainer.get_trained_model()
