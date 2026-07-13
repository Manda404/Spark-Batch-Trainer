Introduction
============

Spark Batch Trainer connects distributed Spark data preparation with local
gradient-boosting libraries. It targets binary and multiclass classification
with XGBoost, CatBoost, and LightGBM.

What the library does
---------------------

1. Assigns a target-stratified batch number to each Spark row.
2. Persists that assignment so the window is not recomputed per batch.
3. Collects the validation DataFrame and one training batch to pandas.
4. Converts categorical columns, reduces pandas memory, and optionally creates
   balanced sample weights.
5. Continues the selected model from the previous batch.
6. Tracks train and validation metrics and applies global early stopping.

What the library does not do
----------------------------

* It does not distribute XGBoost, CatBoost, or LightGBM training through Spark.
* It does not guarantee that a requested number of batches fits driver memory.
* It does not provide complete feature engineering, imputation, or encoding.
* It does not make continuation training equivalent to fitting once on all rows.
* It does not currently expose a common model persistence API.

Continuation semantics
----------------------

Existing trees are retained when the next batch is trained, but older examples
are not revisited when new gradients are computed. Results can therefore depend
on batch order, batch count, and the number of boosting rounds per batch.

Use the library when this trade-off is explicit and validated for your task.
See :doc:`concepts` for the operational details.
