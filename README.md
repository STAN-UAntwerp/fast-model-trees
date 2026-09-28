# fast-model-trees

Fast implementation of **PILOT** (PIecewise Linear Organic Trees) and **RaFFLE** (Random Forest Featuring Linear Extensions) algorithms.

## Overview

This package provides efficient C++-based implementations of:

- **PILOT**: A linear model tree algorithm that builds piecewise linear models
- **RaFFLE**: A random forest ensemble method using PILOT trees as base learners

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

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Development

For development setup, running benchmarks, and reproducing paper results, see [DEVELOPMENT.md](DEVELOPMENT.md).

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

# Inspect the tree structure
print(model.tree_summary())
```

## Key Features

- **Fast C++ implementation**: Optimized for performance using Armadillo linear algebra
- **Scikit-learn compatible**: Follows scikit-learn API conventions
- **Flexible model complexity**: Control tree depth and piecewise linear behavior
- **Feature importance**: Built-in feature importance calculation
- **Bootstrap aggregation**: RaFFLE uses bootstrap sampling for robust predictions

## Parameters

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
- `con_full_search`: If a node's best model on its random feature subset is the constant (con) model, extend the model search to all features of the tree and only make the node a leaf if con remains optimal (default: True). Only has an effect when `n_features_node < n_features_tree`. Set to False to reproduce versions <= 0.1.1.

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