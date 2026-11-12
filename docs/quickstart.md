# Quickstart

## A single PILOT tree

```python
import numpy as np
from pilot import PILOT

rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 10))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(1000)

tree = PILOT(max_depth=5)
tree.train(X, y)            # all features numerical by default

tree.predict(X[:5])
tree.feature_importances_   # normalized RSS reduction per feature
summary = tree.tree_summary(feature_names=[f"x{i}" for i in range(10)])
```

`tree_summary()` returns one row per node, including the node's model, its split, and
the number of training samples that reached it (`n_samples`). See
{doc}`user_guide/pilot` for the details.

## A RaFFLE forest

```python
from pilot import RaFFLE

forest = RaFFLE(n_estimators=100, max_depth=5, random_state=42, oob_score=True)
forest.fit(X, y)

forest.predict(X[:5])
forest.oob_score_            # out-of-bag R², an estimate of the test R²
forest.feature_importances_
```

`RaFFLE` follows the scikit-learn estimator API (`fit` / `predict` / `get_params`),
so it can be used with tools such as {func}`sklearn.model_selection.cross_val_score`.

## Categorical features

Categorical features must be label encoded (integers `0, 1, 2, ...`). Tell the model
which columns are categorical:

```python
tree.train(X, y, categorical=[0, 1, 0, 0, ...])   # PILOT: 0/1 indicator per feature
forest.fit(X, y, categorical_idx=[1])             # RaFFLE: column indices
```

Categorical features are split with `pconc` nodes, which send a subset of the levels to
the left child.
