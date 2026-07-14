# Training Workflow

```mermaid
sequenceDiagram
  actor User
  participant Backend
  participant Spark
  participant Driver
  participant Model
  User->>Backend: fit(train, validation, target, configs)
  Backend->>Backend: validate common configuration
  Backend->>Spark: assign stratified batch numbers
  Backend->>Driver: collect validation data
  loop each batch
    Spark->>Driver: collect one persisted batch
    Driver->>Driver: convert categories, downcast, compute weights
    Backend->>Model: continue from the previous model
    Model-->>Backend: train and validation metric history
    Backend->>Backend: update best model and patience
  end
  Backend-->>User: model and immutable training history
```

Batch assignment is persisted with `MEMORY_AND_DISK` and released in a `finally` block. Global early stopping compares the last configured metric after every batch. Known score metrics are maximized, losses are minimized, and custom metrics can declare an explicit direction.
