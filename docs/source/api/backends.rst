Model Backends
==============

XGBoost
-------

.. autoclass:: spark_batch_trainer.backends.xgboost.XGBoostTrainer
   :members: fit, get_trained_model, get_training_history

CatBoost
--------

.. autoclass:: spark_batch_trainer.backends.catboost.CatBoostTrainer
   :members: fit, get_trained_model, get_training_history

LightGBM
--------

.. autoclass:: spark_batch_trainer.backends.lightgbm.LightGBMTrainer
   :members: fit, get_trained_model, get_training_history
