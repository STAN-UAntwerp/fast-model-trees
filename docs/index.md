# fast-model-trees

Fast C++ implementations of two tree-based regression methods, with a Python API:

**PILOT** (PIecewise Linear Organic Trees)
: A linear model tree. Instead of a constant, every node fits a small linear model
  (piecewise constant, piecewise linear, broken linear, ...) on one feature, and the
  type of model is chosen per node with a BIC criterion. See {doc}`user_guide/pilot`.

**RaFFLE** (Random Forest Featuring Linear Extensions)
: A random forest of PILOT trees, each trained on a bootstrap sample and a random
  subset of the features. See {doc}`user_guide/raffle`.

```python
import numpy as np
from pilot import RaFFLE

rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 10))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(1000)

model = RaFFLE(n_estimators=100, max_depth=5, oob_score=True).fit(X, y)
model.predict(X[:5])
model.oob_score_          # out-of-bag R²
model.feature_importances_
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/pilot
user_guide/raffle
user_guide/feature_importance
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
citing
```

```{toctree}
:caption: Project
:hidden:

GitHub <https://github.com/STAN-UAntwerp/fast-model-trees>
PyPI <https://pypi.org/project/fast-model-trees/>
```
