Tutorials
=========

This section introduces the fundamental concepts of **Spark Batch Trainer**,
essential to understanding its architecture and usage.  

📘 Complete and reproducible use cases (datasets, executable code)
are detailed separately in the :doc:`examples` section.

---

Quickstart
----------

Here is a minimal example to get started with **XGBoostTrainer**:

.. code-block:: python

   from spark_batch_trainer.trainers import XGBoostTrainer

   trainer = XGBoostTrainer()

   config_model={"objective": "binary:logistic", "max_depth": 6},
   config_training={"num_batches": 5, "show_learning_curve": True}
   config_lr_scheduler = {}  # Empty dictionary → use fixed learning rate (no scheduler)

   trainer.fit(spark_train_df, spark_valid_df, config_model, config_training, config_lr_scheduler)

   model = trainer.get_trained_model()

This quickstart shows the basic usage: provide preprocessed Spark DataFrames,
define configurations, fit the model, and retrieve the trained estimator.

---

Defaults
--------

Before diving deeper, it is important to understand the default behavior of the framework:

- Input must always be **preprocessed Spark DataFrames** (train & validation).  
- If a configuration dictionary is left empty (`{}`), Spark Batch Trainer applies its internal defaults.  
- Supported ML frameworks: **XGBoost**, **CatBoost**, **LightGBM**.  

---

1. General Workflow
-------------------

The workflow of **Spark Batch Trainer** follows a standardized process:

1. Prepare preprocessed **Spark DataFrames** (train and validation)  
2. Define the configuration dictionaries  
3. Instantiate a **trainer** for the chosen ML framework  
4. Launch **batch-wise training** and evaluate the trained model  

Workflow illustration:

.. code-block::

   Spark DataFrames
        │
        ├── Train set
        ├── Validation set
        ▼
   [ Spark Batch Trainer ]
        │
        ├── Config dictionaries
        │
        ▼
   [ Trainer (XGBoost / CatBoost / LightGBM) ]
        │
        ▼
   Batch-wise training → Evaluation → Trained Model

.. note::
   The user must provide a **train set** and a **validation set**
   as **preprocessed Spark DataFrames**.

---

2. Data Preparation
-------------------

The framework expects **preprocessed Spark DataFrames** as input.  

.. note::

   Spark Batch Trainer does not perform **full preprocessing**
   (cleaning, encoding, normalization, handling missing values, etc.).  
   These steps must be performed upstream by the user.

   However, the framework provides some automatic features:  

   - Conversion of *object* columns into **categorical**  
   - Memory optimization of Pandas DataFrames  
     (automatic downcasting of integers/floats, conversion to categories)  
   - Detection of categorical columns and automatic adjustment of hyperparameters
     in the underlying model (native support in XGBoost, CatBoost, LightGBM)  

This design ensures that users keep full control over preprocessing,
while still benefiting from some convenient automatic optimizations.

---

3. Configuration Dictionaries
-----------------------------

Training is controlled by three dictionaries.  
Each can be left empty (`{}`) to use default parameters.

### a) `config_model`

This dictionary defines the **hyperparameters of the chosen ML model**
(**XGBoost**, **CatBoost**, **LightGBM**).  

📌 Why define it?

- Adapt the model to the task type (binary, multiclass, regression)  
- Optimize performance (tree depth, regularization, sampling rate, etc.)  
- Guarantee reproducibility through `random_state`  

.. note::

   If this dictionary is left empty (`{}`), the framework applies
   its internal default parameters.

.. warning::

   For **multiclass classification**, it is strongly recommended to
   explicitly define this dictionary to avoid errors or performance
   degradation during training.

### b) `config_training`

This dictionary controls the parameters related to **batch-wise training**.  

📌 Why define it?

- `num_batches` → controls how the dataset is split into manageable parts  
- `max_patience` → defines the global patience for early stopping  
- `show_learning_curve` → enables visualization of train/validation learning curves  
- `use_sample_weight` → activates weighting for imbalanced datasets  

With this configuration, users can balance performance monitoring,
memory efficiency, and control over training convergence.

### c) `config_lr_scheduler`

This dictionary defines the strategy for **dynamic learning rate scheduling**.  

📌 What is a *learning rate scheduler*?  

A *learning rate scheduler* is a mechanism that automatically adjusts the **learning rate**
during training.  

The idea is to begin with a relatively high learning rate, which allows the model to
explore the parameter space quickly when it is still far from the optimum.  
As training progresses and the model approaches convergence, the learning rate is gradually reduced.  
This makes updates more precise, stabilizes optimization, and reduces the risk of overshooting.  

📌 Why use it?

- **Faster convergence** at the beginning of training  
- **Improved stability** as the model approaches the optimum  
- **Reduced overfitting** on noisy or complex datasets  

Example strategy: **Exponential Decay**, with:  

- `initial_lr` → starting learning rate  
- `decay_rate` → reduction factor applied at each step  
- `min_lr` → minimum learning rate threshold  

Alternative strategy: **Step Decay** (not implemented in this version):  
The learning rate is divided by a constant factor every *s* iterations.  

.. note::

   In the current version of **Spark Batch Trainer**, only
   **exponential decay** is implemented.  

   If this dictionary is left empty (`{}`), the model keeps
   its fixed learning rate by default.  

   For further details, see:  
   `Introduction to Learning Rate Schedulers <https://medium.com/@theom/a-very-short-visual-introduction-to-learning-rate-schedulers-with-code-189eddffdb00>`_

---

4. Trainers
-----------

**Spark Batch Trainer** provides several trainer implementations,
each associated with a boosting framework:  

- `XGBoostTrainer` (**XGBoost**) :  
  Flexible and powerful, especially effective for heterogeneous datasets
  and binary classification tasks.  

- `CatBoostTrainer` (**CatBoost**) :  
  Optimized for native handling of **categorical variables**, reducing
  the need for manual preprocessing.  

- `LightGBMTrainer` (**LightGBM**) :  
  Designed for **speed and memory efficiency**, making it highly effective
  for very large datasets with reduced training time.  

All trainers inherit from a common abstract class (`BaseTrainer`)
and share a unified interface:  

- `.fit()` → launches training  
- `.get_trained_model()` → retrieves the trained model  

This modular design ensures a consistent API across frameworks
and allows easy integration of new trainers.

---

5. Batch-wise Training
----------------------

The core of the framework is **batch-wise training**.  

Instead of training the model on the full dataset at once,
the dataset is divided into **successive batches**.  
The model is first trained on the initial batch, then incrementally
refined with each subsequent batch.  

📌 Advantages:  

- **Reduced memory consumption**: only part of the dataset is loaded at a time  
- **Scalability**: supports massive datasets in distributed environments  
- **Better monitoring**: performance can be tracked batch by batch  
- **Global early stopping**: if no improvement is seen within the patience window, training can stop early  

This approach makes the framework highly suited for **Big Data contexts**
and production-scale environments.

---

6. Model Evaluation and Retrieval
---------------------------------

At the end of training, the user can:  

- Retrieve the trained model with `.get_trained_model()`  
- Make predictions on new datasets  
- Visualize learning curves to assess performance  
- Save or export the model using the native methods of the underlying framework  
  (e.g., `.save_model()` for XGBoost or LightGBM)  

Evaluation is performed using the **test dataset defined upstream**, ensuring
an unbiased measure of generalization.

---

7. Best Practices
-----------------

To make the most of **Spark Batch Trainer**, it is recommended to:  

- Carefully check your splits (train/valid/test) to avoid data leakage  
- Explicitly define `config_model` for both binary and multiclass tasks  
- Enable `show_learning_curve=True` in `config_training` to monitor overfitting  
- Use `use_sample_weight` when working with imbalanced datasets  
- Save models using the native methods of the underlying ML framework  

---

8. Limitations
--------------

- No automatic preprocessing (cleaning, encoding, normalization, missing values).  
- No built-in handling of train/validation/test splits.  
- Only **exponential decay** is implemented for the learning rate scheduler.  

---

📘 For complete use cases (datasets, reproducible code, results), see the :doc:`examples` section.
