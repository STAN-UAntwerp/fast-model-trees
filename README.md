# fast-model-trees

[![Documentation](https://readthedocs.org/projects/fast-model-trees/badge/?version=latest)](https://fast-model-trees.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/fast-model-trees)](https://pypi.org/project/fast-model-trees/)

Fast implementation of **PILOT** (PIecewise Linear Organic Trees) and **RaFFLE** (Random Forest Featuring Linear Extensions) algorithms.

## Overview

This package provides efficient C++-based implementations of:

- **PILOT**: A linear model tree algorithm that builds piecewise linear models
- **RaFFLE**: A random forest ensemble method using PILOT trees as base learners

📖 **Documentation: <https://fast-model-trees.readthedocs.io>**: installation, user guide and API reference.

## Papers

- **PILOT**: Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT. *Machine Learning*, 1-50. https://doi.org/10.1007/s10994-024-06590-3

- **RaFFLE**: Raymaekers, J., Rousseeuw, P. J., Servotte, T., Verdonck, T., & Yao, R. (2025). A Powerful Random Forest Featuring Linear Extensions (RaFFLE). *Under Review*

## Installation

```bash
pip install fast-model-trees
```

### Building from Source

If you need to build from source, you'll need:
- C++17 compatible compiler
- CMake >= 3.12
- Armadillo linear algebra library
- BLAS and LAPACK libraries
- pybind11
- carma (C++ Armadillo/NumPy bridge)

See the [installation guide](https://fast-model-trees.readthedocs.io/page/installation.html) for platform-specific instructions.

### Development

For development setup, running benchmarks, and reproducing paper results, see [DEVELOPMENT.md](https://github.com/STAN-UAntwerp/fast-model-trees/blob/main/DEVELOPMENT.md).

## Quick Start

### Using RaFFLE (Random Forest)

```python
from pilot import RaFFLE
import numpy as np

# Create sample data
X = np.random.randn(1000, 10)
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(1000) * 0.1

# Train RaFFLE model
model = RaFFLE(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Get feature importances
importances = model.feature_importances_

# Out-of-bag estimate of the test R² (set oob_score=True when constructing the model)
model = RaFFLE(n_estimators=100, max_depth=5, oob_score=True).fit(X, y)
model.oob_score_
model.oob_predict(X)        # OOB predictions of the training rows, any metric
model.estimators_samples_   # bootstrap indices of every tree
```

### Using PILOT (Single Tree)

```python
from pilot import PILOT
import numpy as np

# Create sample data
X = np.random.randn(1000, 10)
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(1000) * 0.1

# Train PILOT model — all parameters have sensible defaults
model = PILOT(max_depth=5)
model.train(X, y)  # categorical defaults to all-numerical

# Make predictions
predictions = model.predict(X)

# Inspect the tree structure (one row per node, incl. the number of training samples)
summary = model.tree_summary()
leaf_sizes = summary.loc[summary["node_type"] == "con", ["node_id", "n_samples"]]
```

## Key Features

- **Fast C++ implementation**: Optimized for performance using Armadillo linear algebra
- **Scikit-learn compatible**: Follows scikit-learn API conventions
- **Flexible model complexity**: Control tree depth and piecewise linear behavior
- **Feature importance**: Built-in feature importance calculation
- **Bootstrap aggregation**: RaFFLE uses bootstrap sampling for robust predictions
- **Out-of-bag estimates**: bootstrap indices are stored per tree, for OOB predictions and scores
- **Tree inspection**: `tree_summary()` lists every node with its model, split and sample count

## Parameters

The most important parameters are listed below; see the [API reference](https://fast-model-trees.readthedocs.io/page/api.html) for all of them, and the user guides for [PILOT](https://fast-model-trees.readthedocs.io/page/user_guide/pilot.html) and [RaFFLE](https://fast-model-trees.readthedocs.io/page/user_guide/raffle.html) for how they work.

### RaFFLE

- `n_estimators`: Number of trees in the forest (default: 10)
- `max_depth`: Maximum depth of each tree (default: 12)
- `min_sample_fit`: Minimum samples needed to fit any node (default: 10)
- `min_sample_alpha`: Minimum samples for piecewise nodes (default: 5)
- `min_sample_leaf`: Minimum samples in each leaf (default: 5)
- `random_state`: Random seed for reproducibility (default: 42)
- `n_features_tree`: Fraction of features to consider per tree (default: 1.0)
- `n_features_node`: Fraction of features to consider per node (default: 1.0)
- `alpha`: Controls piecewise linear complexity (default: 1)
- `oob_score`: Compute the out-of-bag score during `fit`: `True` for R², or a metric `f(y_true, y_pred)` (default: False)

### PILOT

- `df_settings`: Degrees of freedom for node types `[con, lin, pcon, blin, plin, pconc]`. Defaults to `[1, 2, 5, 5, 7, 5]`. Set a value to `-1` to disable that node type.
- `max_depth`: Maximum split depth (default: 20)
- `max_model_depth`: Maximum total depth including linear model nodes (default: 100)
- `max_features`: Features considered per split; `None` (default) uses all features
- `min_sample_fit`: Minimum samples to fit any node (default: 5)
- `min_sample_alpha`: Minimum samples for piecewise nodes (default: 5)
- `min_sample_leaf`: Minimum samples per leaf (default: 5)
- `rel_tolerance`: Minimum relative RSS improvement to keep growing (default: 0.01)

## License

MIT License - see LICENSE file for details

## Citation

If you use this package in your research, please cite:

```bibtex
@article{raymaekers2024pilot,
  title={Fast linear model trees by PILOT},
  author={Raymaekers, J. and Rousseeuw, P.J. and Verdonck, T. and Yao, R.},
  journal={Machine Learning},
  pages={1--50},
  year={2024},
  doi={10.1007/s10994-024-06590-3}
}
```

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/STAN-UAntwerp/fast-model-trees) for more information.

## Support

For issues and questions, please use the [GitHub issue tracker](https://github.com/STAN-UAntwerp/fast-model-trees/issues).