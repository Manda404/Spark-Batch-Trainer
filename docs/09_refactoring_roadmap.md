# Refactoring Roadmap

## Completed

1. source audit and characterization tests;
2. metric direction and XGBoost continuation fixes;
3. immutable common configuration;
4. shared evaluation policy;
5. Spark batching and pandas preparation extraction;
6. backend file renaming with compatibility shims;
7. lazy imports, optional plotting, and opt-in file logging;
8. English naming and documentation pass.

## Next

1. introduce a typed runtime state and `TrainingResult`;
2. add pre-collection memory budgets;
3. extract one shared orchestration loop;
4. normalize categorical schemas;
5. enforce a global boosting-round budget;
6. implement native save/load per backend;
7. add Spark plan and high-volume memory benchmarks.

Every stage must preserve binary and multiclass integration tests for all three backends.
