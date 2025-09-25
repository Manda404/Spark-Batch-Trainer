Introduction
============

**Spark Batch Trainer** is a modular and extensible framework dedicated to
**batch-wise training** of boosting models in machine learning,
with a strong focus on memory efficiency and scalability.

---

What is Batch-wise Training?
----------------------------

*Batch-wise training* means splitting large datasets into smaller chunks
(*batches*). Instead of loading all the data into memory at once, the model
is trained incrementally on each batch.  

This approach is especially useful when:

- The dataset is too large to fit into memory.  
- The classes are imbalanced and must be represented in every training step.  
- Incremental learning and reproducibility are required in Big Data contexts.  

---

How Spark Batch Trainer Works
-----------------------------

**Spark Batch Trainer** is designed to integrate seamlessly with:  

- **Apache Spark DataFrames**  
  → Distributed and scalable preprocessing, so even very large datasets can be handled efficiently.  

- **Popular ML frameworks**  
  → Native integration with **XGBoost**, **CatBoost**, and **LightGBM**,  
  optimized specifically for boosting models.  

- **Big Data scenarios**  
  → Supports incremental learning, class balancing, and real-time monitoring of **learning curves**.  

- **Target use case**  
  → This initial release focuses on **classification tasks** (binary and multiclass).  

---

Typical Workflow
----------------

Training a model with Spark Batch Trainer can be summarized in three simple steps:

1. **Provide two Spark DataFrames**: one for training (**train set**) and one for validation (**validation set**).  
2. **Define your configurations**: model parameters, batch-wise training strategy, and (optionally) a learning rate scheduler.  
3. **Train incrementally** with monitoring tools such as learning curves and confusion matrices.  

---

Key Features
------------

- **Stratified batch creation**  
  Ensures balanced class distribution in each batch, particularly important for imbalanced datasets.  

- **Seamless Spark ↔ Pandas conversion**  
  Switch easily between distributed Spark DataFrames and local Pandas DataFrames, with automatic type conversion (e.g. *object* → *category*).  

- **Built-in visualization**  
  Monitor learning curves, learning rate schedules, and confusion matrices to better interpret model performance.  

- **Extensibility**  
  Add new trainers or custom learning rate schedulers with minimal effort.  

---

Why Spark Batch Trainer?
------------------------

- **Scalable**: Handles datasets from MBs to TBs.  
- **Modular**: Works seamlessly with multiple boosting frameworks.  
- **Reproducible**: Same workflow can be replayed across experiments and production.  
- **Transparent**: Visual monitoring tools provide insight into training dynamics.  

---

**Goal**: provide data scientists with a simple, reproducible, and extensible tool  
to train boosting models at scale.  
