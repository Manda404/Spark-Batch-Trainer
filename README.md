# SparkBatchTrainer


This project provides a scalable and modular framework for training machine learning models in **batches** using **Spark DataFrames**.   It is designed to support large datasets, incremental learning, and integration with **XGBoost**, **CatBoost**, and **LightGBM**.


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue.svg)](https://python-poetry.org/)

---

## 📑 Table of Contents

- [Project Goal](#-project-goal)
- [Problem Addressed](#-problem-addressed)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)

---

## 🎯 Project Goal

SparkBatchTrainer is a framework designed to train machine learning models on **very large distributed datasets**. It combines the power of **Apache Spark** for data processing with the efficiency of **gradient boosting algorithms** (XGBoost, CatBoost, LightGBM) to create an **incremental batch-wise training system** that is both reproducible and extensible.

---

## 🔍 Problem Addressed

- **Massive datasets**: Impossible to load the entire dataset into memory
- **Sequential training**: Need to train on coherent data subsets
- **Continuous optimization**: Dynamic adjustment of hyperparameters (`learning_rate`, `early_stopping`)
- **Class balancing**: Maintaining target distribution across batches

---

## 🏗️ Architecture

The framework is structured around several core components:

```
src/
└── spark_batch_trainer/
    │
    ├── __init__.py
    │
    ├── core/
    │   ├── __init__.py                    # Exposes OptimizedWeightCalculator and MemoryOptimizer
    │   ├── base_trainer.py                # Abstract class: fit(), get_trained_model(), save(), load()
    │   ├── class_weight_optimizer.py      # OptimizedWeightCalculator (cache, smoothing, normalization)
    │   └── memory_optimizer.py            # MemoryOptimizer (Pandas DataFrame memory optimization)
    │
    ├── trainers/
    │   ├── __init__.py
    │   ├── xgboost_trainer.py             # Batch-wise XGBoost training with class weights
    │   ├── catboost_trainer.py            # Batch-wise CatBoost training with class weights
    │   └── lightgbm_trainer.py            # Batch-wise LightGBM training with learning rate scheduling
    │
    └── logger/
        └── logger.py                      # Centralized logger (handlers, levels, formatting)
```

---

## ⚡ Key Features

### 1. **Incremental Batch Training**
- **Automatic stratification**: Each batch preserves class distribution
- **Randomized ordering**: Reproducible shuffling with a seed
- **Warm restart**: Training continuity across batches

### 2. **Advanced Optimization**
- **Learning Rate Scheduling**: Exponential decay or custom schedulers
- **Global Early Stopping**: Based on inter-batch validation performance
- **Class balancing**: Automatic sample weight calculation with `OptimizedWeightCalculator`

### 3. **Monitoring and Visualization**
- **Learning curves**: Training/validation metrics visualization
- **Dynamic hyperparameters**: Track learning rate evolution per batch
- **Detailed logging**: Full training traceability

### 4. **Multi-Model Support**
- **XGBoostTrainer**
- **CatBoostTrainer**
- **LightGBMTrainer**

---

## 📦 Installation

### Requirements

- Python 3.8+
- Poetry (dependency management)
- Apache Spark 3.0+
- PySpark
- XGBoost
- CatBoost
- LightGBM
- NumPy
- Pandas

### Install Poetry

If you don't have Poetry installed:

```bash
# Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Or with pip
pip install poetry
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/Manda404/SparkBatchTrainer.git
cd SparkBatchTrainer

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Install in production mode (without dev dependencies)

```bash
poetry install --only main
```

### Verify installation

```bash
# Check installed packages
poetry show

# Run a quick test
poetry run python -c "from spark_batch_trainer.trainers import XGBoostTrainer; print('Installation successful!')"
```

---

## 🚀 Quick Start

### Basic Usage with XGBoostTrainer

```python
from pyspark.sql import SparkSession
from spark_batch_trainer import XGBoostTrainer

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BatchTrainingExample") \
    .getOrCreate()

# Load your data
spark_train_df = spark.read.parquet("path/to/train_data.parquet")
spark_valid_df = spark.read.parquet("path/to/valid_data.parquet")

# Initialize trainer
trainer = XGBoostTrainer()

# Train model
model = trainer.fit(
    train_dataframe=spark_train_df,
    valid_dataframe=spark_valid_df,
    target_column="NObeyesdad",
    config_model={
        "objective": "multi:softprob",
        "n_estimators": 100,
        "num_class": 7,
        "max_depth": 6,
        "eval_metric": "mlogloss"
    },
    config_training={
        "num_batches": 5,              # Number of training batches
        "max_patience": 5,             # Patience for global early stopping
        "show_learning_curve": True,   # Display learning curves
    },
    config_lr_scheduler={
        "initial_lr": 0.1,
        "decay_rate": 0.95
    }
)

# Save trained model
trainer.save("models/xgboost_model.pkl")

# Load model later
loaded_trainer = XGBoostTrainer()
loaded_trainer.load("models/xgboost_model.pkl")
```

### Using CatBoostTrainer

```python
from spark_batch_trainer import CatBoostTrainer

trainer = CatBoostTrainer()
model = trainer.fit(
    train_dataframe=spark_train_df,
    valid_dataframe=spark_valid_df,
    target_column="target",
    config_model={
        "iterations": 100,
        "depth": 6,
        "loss_function": "MultiClass",
        "classes_count": 7
    },
    config_training={
        "num_batches": 5,
        "max_patience": 3,
        "show_learning_curve": True
    }
)
```

### Using LightGBMTrainer

```python
from spark_batch_trainer import LightGBMTrainer

trainer = LightGBMTrainer()
model = trainer.fit(
    train_dataframe=spark_train_df,
    valid_dataframe=spark_valid_df,
    target_column="target",
    config_model={
        "objective": "multiclass",
        "num_class": 7,
        "num_iterations": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    config_training={
        "num_batches": 5,
        "max_patience": 5,
        "show_learning_curve": True
    },
    config_lr_scheduler={
        "initial_lr": 0.1,
        "decay_rate": 0.9
    }
)
```

---
