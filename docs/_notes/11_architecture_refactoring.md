# Architecture Refactoring

## Implemented structure

```text
spark_batch_trainer/
├── __init__.py
├── factory.py
├── logging.py
├── backends/
├── data/
└── training/
```

Training components do not depend on model SDKs. Plotting imports Matplotlib
only when requested. Public backend exports are lazy, and Spark batch
assignment is persisted once.

Redundant compatibility packages and generated log files were removed. The
package now has three functional areas instead of parallel old and new trees.
The public root remains the recommended import path.

## Remaining work

`BatchTrainer` and the model backends are still large. Their orchestration
should be decomposed only when contract tests can protect continuation
training semantics across all three SDKs. Splitting them merely to reduce line
counts would recreate the unnecessary-file problem.
