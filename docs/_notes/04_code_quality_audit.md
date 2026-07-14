# Python Code Quality Audit

## Improvements already delivered

- configuration validation moved to an immutable dataclass;
- metric comparison and early stopping extracted from model backends;
- Spark batching, pandas preparation, memory optimization, weighting, plotting, and logging now have explicit modules;
- model dependencies and Matplotlib are imported lazily;
- generated log files are opt-in;
- public history is immutable;
- old imports are isolated in compatibility shims.

## Remaining debt

- backend classes remain large and duplicate validation/preparation code;
- runtime state still uses `Dict[str, Any]`;
- model persistence is absent;
- categorical schemas are inferred independently across collections;
- `deepcopy` snapshots can increase memory pressure;
- the first SDK metric is still selected when multiple metrics are returned.

Dataclasses remain the preferred choice for internal state. Pydantic would only be justified at a JSON or service boundary. A small backend protocol should be introduced only when the orchestration loop is extracted.
