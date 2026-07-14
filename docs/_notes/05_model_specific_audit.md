# Model-Specific Audit

## XGBoost — fragile but improved

Continuation uses `xgb_model` from the preceding batch. The earlier mismatch between the current estimator and the best snapshot was removed. Every batch still adds another `n_estimators` trees, so a global tree budget is required.

## CatBoost — generally correct with reservations

Each batch creates a classifier and continues with `init_model`. Pools, categorical features, and optional weights are supported. The result remains sensitive to batch order, and no best iteration inside a batch is restored.

## LightGBM — generally correct with reservations

Each batch continues with `init_model`. The configured learning rate is preserved when no scheduler is supplied. Every batch adds `n_estimators`; no intra-batch callback currently limits that budget.

For all backends, this is continuation training rather than online learning. Previous trees remain, but gradients from older rows are not recomputed.
