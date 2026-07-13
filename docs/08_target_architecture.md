# Target Architecture

The target remains intentionally small:

```text
public facade
  -> shared training orchestrator
     -> validated configuration
     -> Spark batch source
     -> pandas preparation and weighting
     -> backend adapter
     -> evaluation policy
     -> training result and model store
```

The next structural step is a typed `TrainingRunState` plus a shared orchestration loop. A backend adapter should only know how to create, continue, evaluate, and serialize its SDK model. Avoid one-line interfaces and configuration folders with no independent responsibility.

Persistence must use each SDK's native format plus a versioned manifest containing backend, feature schema, categorical metadata, configuration, metric direction, and package versions.
