# English Naming Conventions

## Preferred vocabulary

| Avoid | Use |
|---|---|
| `config_model` | `model_config` |
| `config_training` | `training_config` at the API boundary |
| mutable training config | `runtime_state` inside a training run |
| `config_lr_scheduler` | `learning_rate_config` |
| `processed_batch` | `pandas_batch` |
| `valid_data` | `validation_data` |
| `current_lr` | `learning_rate` |
| generic `_optimizer` | `_memory_optimizer` |
| generic `_weight_calculator` | `_sample_weight_calculator` |
| `MemoryOptimizer` | `PandasMemoryOptimizer` |
| `OptimizedWeightCalculator` | `BalancedSampleWeightCalculator` |

## Method naming

Methods use a verb followed by a concrete object: `iter_training_batches`, `collect_validation_data`, `convert_categorical_features`, `extract_metric_history`, and `calculate_scheduled_learning_rate`.

Public compatibility aliases are retained for one migration period. New implementation code must import from `backends`, `data`, `evaluation`, `config`, `visualization`, or `observability` rather than legacy shim packages.
