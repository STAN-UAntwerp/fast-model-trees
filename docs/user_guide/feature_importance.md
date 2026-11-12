# Feature importance

Both {class}`~pilot.PILOT` and {class}`~pilot.RaFFLE` expose a `feature_importances_`
attribute, computed like the impurity-based importances of scikit-learn's random forests.

## Single tree

While a tree grows, every `lin` and split node adds its RSS reduction (the RSS of the
node's samples before and after fitting its model) to the feature it uses. These totals
are normalized to sum to 1:

```python
tree = PILOT(max_depth=5).train(X, y)
tree.feature_importances_
```

The per-node RSS reductions are the `rss_reduction` column of
{meth}`~pilot.PILOT.tree_summary`, so the importances can be recomputed, or broken down
by depth or node type, from the summary.

## Forest

For a RaFFLE forest:

1. each tree's importances are normalized to sum to 1, so that every tree contributes
   equally whatever the scale of its RSS,
2. the normalized importances are averaged across the trees (a feature a tree was not
   trained on counts as 0 for that tree),
3. the average is normalized to sum to 1.

```python
forest = RaFFLE(n_estimators=100).fit(X, y)
forest.feature_importances_                 # shape (n_features,)
forest.estimators[0].feature_importances_   # one tree, over its own feature_idx
```

## Interpretation

As with all impurity-based importances, the values are computed on the training data
and favour features that offer many split points. For an importance measured on held-out
data, use {func}`sklearn.inspection.permutation_importance`, which works with `RaFFLE`
directly.

Because PILOT nodes fit linear models, a feature with a linear effect gets its importance
from `lin` or `plin` nodes, where a CART tree would need many constant splits to capture
it. Importances of PILOT and CART can therefore rank features differently on the same
data.
