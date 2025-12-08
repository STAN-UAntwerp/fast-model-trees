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

See the full [installation guide](https://github.com/STAN-UAntwerp/fast-model-trees) for detailed instructions.

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
from pilot import PILOT, DEFAULT_DF_SETTINGS
import numpy as np

# Create sample data
X = np.random.randn(1000, 10)
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(1000) * 0.1

# Train PILOT model
model = PILOT(
    df_settings=list(DEFAULT_DF_SETTINGS.values()),
    max_depth=5,
    max_features=X.shape[1]
)

# Categorical features (0 = numerical, 1 = categorical)
categorical = np.zeros(X.shape[1], dtype=int)

model.train(X, y, categorical)

# Make predictions
predictions = model.predict(X)
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

### PILOT

- `df_settings`: Degrees of freedom for different node types
- `max_depth`: Maximum tree depth
- `max_features`: Maximum features to consider
- `min_sample_fit`: Minimum samples for fitting
- `min_sample_alpha`: Minimum samples for piecewise splits
- `min_sample_leaf`: Minimum samples per leaf

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