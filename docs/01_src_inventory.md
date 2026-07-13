# Current Source Inventory

```text
spark_batch_trainer/
├── factory.py                 # public factory
├── logging.py                 # logging configuration
├── backends/                  # XGBoost, CatBoost, LightGBM adapters
├── data/                      # Spark batching and pandas preparation
└── training/                  # lifecycle, configuration, metrics and history
```

The recommended public entry points are `create_trainer`, `XGBoostTrainer`, `CatBoostTrainer`, and `LightGBMTrainer`. Model SDKs are loaded lazily. Redundant compatibility modules have been removed.

External runtime dependencies are PySpark, pandas, NumPy, scikit-learn, Matplotlib, XGBoost, CatBoost, and LightGBM. No static circular dependency was detected.
