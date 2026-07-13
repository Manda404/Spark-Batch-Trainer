Practical Training Guide
========================

1. Prepare compatible Spark DataFrames
--------------------------------------

Training and validation DataFrames must contain the target and identical
feature columns. Perform null handling, domain validation, and target encoding
upstream. Keep an independent test set for final evaluation.

2. Estimate driver memory
-------------------------

The validation set is collected in full. Estimate its pandas size before
training and choose a batch count that leaves room for temporary arrays and
multiple model snapshots. ``num_batches`` is not a byte limit.

3. Select one backend
---------------------

Use ``create_trainer()`` for a simple runtime choice or import a trainer class
directly for static code.

.. code-block:: python

   from spark_batch_trainer import LightGBMTrainer

   trainer = LightGBMTrainer()

4. Configure stopping explicitly
--------------------------------

Use ``metric_mode="max"`` for score metrics and ``metric_mode="min"`` for
losses when automatic detection is not appropriate.

.. code-block:: python

   training_config = {
       "num_batches": 8,
       "max_patience": 3,
       "metric_mode": "max",
       "min_delta": 0.001,
       "use_sample_weight": True,
   }

5. Inspect the result
---------------------

.. code-block:: python

   model = trainer.get_trained_model()
   history = trainer.get_training_history()

   for batch_number, scores in zip(
       history.batch_numbers,
       history.validation_scores,
   ):
       print(batch_number, scores[-1])

6. Evaluate once on the test set
--------------------------------

Do not select thresholds or hyperparameters on the test set. Convert only a
bounded test sample to pandas, or use a separate distributed inference path.

Operational checklist
---------------------

* Pin Spark and model SDK versions.
* Record model, training, and learning-rate configurations.
* Monitor driver RSS and Spark shuffle spill.
* Confirm categorical schemas are consistent.
* Validate sensitivity to batch order and batch count.
* Use the backend's native persistence API until a shared store is available.
