# Spark and Memory Audit

## Verdict

The complete training dataset may exceed driver memory, but the validation set, one pandas batch, sample weights, temporary copies, and model snapshots must fit simultaneously. The number of batches alone does not guarantee that condition.

Spark performs a partitioned window, random ordering, `ntile`, persistence, filtering, and collection. Training itself is local. The window can cause a shuffle and sort. Assignment is now materialized once and released reliably.

## Remaining safeguards

- estimate rows and bytes before every `toPandas()` call;
- reject collections above a configurable driver budget;
- record driver RSS, Arrow peak memory, shuffle read/write, spill, skew, and model size;
- make deterministic assignment depend on a stable row key rather than random ordering alone;
- enforce one feature schema and categorical vocabulary across validation and batches.
