# PILOT trees

PILOT (PIecewise Linear Organic Trees) is a linear model tree: every node fits a simple
model on a single feature to the residuals left by the nodes above it, and a
prediction is the sum of the contributions of the nodes on the path from the root to a
leaf.

## Node types

At every node PILOT tries every candidate feature with each of the node types below,
and keeps the combination with the lowest BIC

$$
\mathrm{BIC} = n \log\left(\frac{\mathrm{RSS}}{n}\right) + \mathrm{df} \cdot \log(n),
$$

where $n$ is the number of samples in the node and df the degrees of freedom of the
node type.

| Type | Model | Children | Default df |
|------|-------|----------|-----------:|
| `con` | constant | none (leaf) | 1 |
| `lin` | linear model $a + b x$ on all samples | 1 | 2 |
| `pcon` | piecewise constant: split $x \le s$, a constant on each side | 2 | 5 |
| `blin` | broken linear: two lines that meet at the split | 2 | 5 |
| `plin` | piecewise linear: a separate line on each side | 2 | 7 |
| `pconc` | piecewise constant on a categorical feature: a subset of levels goes left | 2 | 5 |

The degrees of freedom are set with `df_settings`, a list in the order
`[con, lin, pcon, blin, plin, pconc]` (see {data}`pilot.DEFAULT_DF_SETTINGS`). A higher
value penalizes a node type more; `-1` disables it:

```python
from pilot import PILOT

# no broken-linear nodes
tree = PILOT(df_settings=[1, 2, 5, -1, 7, 5])
```

## Growing and stopping

- `max_depth` limits the number of splits (`pcon`, `blin`, `plin`, `pconc`) on a path.
  `lin` nodes do not count.
- `max_model_depth` limits the number of nodes on a path, `lin` nodes included.
- `min_sample_fit`: a node with this many samples or fewer becomes a `con` leaf.
- `min_sample_alpha`: piecewise models (splits) are only tried in nodes with at least
  this many samples.
- `min_sample_leaf`: a split is only allowed if both children get at least this many
  samples.
- `rel_tolerance`: a `lin` node must reduce the RSS of its samples by at least this
  fraction; otherwise it is replaced by a `con` leaf.
- `max_features`: number of features, drawn at random, considered at each node.
- `max_pivot`: limit the number of candidate split points per feature to speed up
  training on large data. Cannot be combined with `blin` nodes, so disable them with
  `df_settings[3] = -1`.

To keep the extrapolation of the linear models in check, during prediction the feature
value is clipped to the range seen in the node during training, and the running
prediction is clipped to the range of the training response.

## Inspecting a tree

{meth}`~pilot.PILOT.tree_summary` returns a {class}`pandas.DataFrame` with one row per
node, in pre-order (a node, then its left subtree, then its right subtree):

```python
tree = PILOT(max_depth=3).train(X, y)
df = tree.tree_summary(feature_names=feature_names)
```

| Column | Meaning |
|--------|---------|
| `depth`, `model_depth` | number of splits / number of nodes above the node |
| `node_id`, `parent_node_id` | identifiers to rebuild the tree (`"<model_depth>_<k>"`) |
| `node_type` | `con`, `lin`, `pcon`, `blin`, `plin` or `pconc` |
| `feature_index`, `feature_name` | feature used by the node's model |
| `split_value` | samples with `x <= split_value` go left (`pcon`/`blin`/`plin`) |
| `pivot_values` | levels that go left (`pconc`) |
| `intercept_left`, `slope_left`, `intercept_right`, `slope_right` | coefficients of the node's model on each side |
| `rss_reduction` | RSS reduction achieved by the node (used for feature importance) |
| `n_samples` | number of training samples that reached the node |

See the {meth}`API reference <pilot.PILOT.tree_summary>` for the exact definition of
every column.

### Samples per leaf

The leaves of a PILOT tree are its `con` nodes, so the number of training samples in
each leaf is

```python
leaves = df[df["node_type"] == "con"]
leaves[["node_id", "n_samples"]]
```

The counts of all leaves add up to the number of training samples. A `lin` node passes
all its samples to its single child, and the two children of a split node share the
samples of their parent, so `n_samples` of a node always equals the sum over its
children.
