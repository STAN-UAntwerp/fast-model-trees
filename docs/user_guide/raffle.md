# RaFFLE forests

RaFFLE (Random Forest Featuring Linear Extensions) is a random forest with PILOT trees as
base learners. Each tree is trained on

- a **bootstrap sample** of the training rows: `n_samples` rows drawn with replacement,
- a random subset of the **features**: `n_features_tree` (a fraction, or `"sqrt"`),
  drawn once per tree.

At every node a tree additionally considers a random subset of `n_features_node` of its
features. The forest predicts the average of its trees.

```python
from pilot import RaFFLE

forest = RaFFLE(
    n_estimators=100,
    max_depth=12,
    n_features_tree=1.0,
    n_features_node="sqrt",
    random_state=42,
).fit(X, y)
```

## Model complexity: `alpha`

`alpha` (between 0 and 1) scales the degrees of freedom of the node types relative to
the PILOT defaults: `df = 1 + alpha * (DEFAULT_DF_SETTINGS - 1)`. With `alpha=0` all node
types cost the same as a constant, so the trees use linear and piecewise linear models
freely; with `alpha=1` you get the PILOT defaults. Pass `df_settings` to set the degrees
of freedom directly instead.

## The trees of the forest

The fitted trees are in `forest.estimators`. Each is a {class}`~pilot.PILOTWrapper` with

- `feature_idx`: the columns of `X` the tree was trained on,
- `bootstrap_idx`: the rows of `X` in its bootstrap sample,
- {meth}`~pilot.PILOTWrapper.tree_summary`, {attr}`~pilot.PILOTWrapper.feature_importances_`
  and `predict`, like a single PILOT tree.

```python
tree = forest.estimators[0]
tree.tree_summary(feature_names=feature_names)
forest.predict(X, individual=True)   # one column per tree
```

Because a tree is trained on a bootstrap sample, `n_samples` in its tree summary counts
an observation that was drawn several times several times; the root has
`len(tree.bootstrap_idx)` samples.

## Out-of-bag (OOB) estimates

On average about 37% of the training rows ($(1 - 1/n)^n \approx e^{-1}$) are not in the
bootstrap sample of a given tree. These rows are *out of bag* for that tree, and
predicting them with only the trees that did not see them gives an estimate of the
generalization error without a separate validation set.

### OOB score during `fit`

Set `oob_score=True` to compute the OOB R² at the end of {meth}`~pilot.RaFFLE.fit`:

```python
forest = RaFFLE(n_estimators=100, oob_score=True).fit(X, y)
forest.oob_score_        # OOB R²
forest.oob_prediction_   # OOB prediction of every training row
```

Pass a metric `score(y_true, y_pred)` instead of `True` to use it for `oob_score_`:

```python
from sklearn.metrics import mean_absolute_error

forest = RaFFLE(n_estimators=100, oob_score=mean_absolute_error).fit(X, y)
forest.oob_score_        # OOB mean absolute error
```

With few trees some rows can be in the bootstrap sample of every tree. These rows have no
OOB prediction (NaN in `oob_prediction_`), are left out of `oob_score_`, and `fit`
emits a warning.

### OOB predictions after `fit`

The bootstrap indices are always stored, so the OOB predictions can also be computed
after fitting, on the same training data:

```python
import numpy as np
from sklearn.metrics import mean_squared_error

oob = forest.oob_predict(X)          # X must be the training data, rows in the same order
has_oob = ~np.isnan(oob)
mean_squared_error(y[has_oob], oob[has_oob])
```

For custom analyses, {attr}`~pilot.RaFFLE.estimators_samples_` gives the bootstrap
indices of every tree and {meth}`~pilot.RaFFLE.oob_mask` a boolean matrix with entry
`(i, j)` set when row `i` is out of bag for tree `j`:

```python
mask = forest.oob_mask()                    # (n_samples, n_estimators)
per_tree = forest.predict(X, individual=True)

# OOB error of each individual tree
tree_mse = [
    mean_squared_error(y[mask[:, j]], per_tree[mask[:, j], j])
    for j in range(mask.shape[1])
]
```
