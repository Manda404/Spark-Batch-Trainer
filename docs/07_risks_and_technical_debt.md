# Risk and Technical Debt Register

| ID | Severity | Risk | Current status |
|---|---|---|---|
| R01 | Critical | Wrong metric direction selects the wrong model | Fixed |
| R02 | Critical | A pandas collection can exhaust driver memory | Open |
| R03 | Critical | Inconsistent XGBoost continuation chain | Fixed |
| R04 | High | Spark window recomputed for every batch | Fixed through persistence |
| R05 | High | Total tree count grows by rounds per batch | Open |
| R06 | High | Categorical schemas can differ between collections | Open |
| R07 | High | Repeated `fit()` retains diagnostic state | Fixed |
| R08 | High | Random ordering is not strongly deterministic | Open |
| R09 | High | No public model persistence contract | Open |
| R10 | High | Default learning rate overwrites model configuration | Fixed |
| R11 | Medium | Weights use batch-local distributions | Open |
| R12 | Medium | First returned metric is selected implicitly | Open |
| R13 | Medium | Import writes logs inside the package | Fixed |
| R14 | Medium | Runtime state is an untyped dictionary | Open |
| R15 | Medium | Deep model copies increase peak memory | Open |
| R16 | Medium | Float downcasting can change precision | Open |

The next critical engineering task is R02: enforce a measurable driver-memory collection budget.
