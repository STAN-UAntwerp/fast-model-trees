# Installation Guide for fast-model-trees

## Quick Install (PyPI)

For most users, installation from PyPI is recommended:

```bash
pip install fast-model-trees
```

**Note**: This requires system dependencies (Armadillo, BLAS, LAPACK, carma) to be installed first. See platform-specific instructions below.

## System Dependencies

The package requires the following system libraries:

- **Armadillo**: C++ linear algebra library
- **BLAS/LAPACK**: Linear algebra backends
- **carma**: C++ bridge between Armadillo and NumPy
- **CMake**: Build system (>= 3.12)
- **C++17 compiler**: GCC, Clang, or MSVC

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake g++ libopenblas-dev liblapack-dev libarmadillo-dev
```

Then install carma:
```bash
git clone https://github.com/RUrlus/carma.git
cd carma
mkdir build && cd build
cmake -DCARMA_INSTALL_LIB=ON ..
cmake --build . --config Release --target install
```

### macOS

1. Install Homebrew if not already installed: https://brew.sh/

2. Install dependencies:
```bash
brew install cmake openblas lapack armadillo
```

3. Install carma:
```bash
git clone https://github.com/RUrlus/carma.git
cd carma
mkdir build && cd build
cmake -DCARMA_INSTALL_LIB=ON ..
cmake --build . --config Release --target install
```

### Windows

Windows support is experimental. You'll need:
- Visual Studio 2019 or later with C++ development tools
- CMake (from cmake.org or via chocolatey)
- Install Armadillo and dependencies manually

## Install from Source

If you need the latest development version or want to contribute:

```bash
# Clone the repository
git clone https://github.com/yourusername/fast-model-trees.git
cd fast-model-trees

# Install system dependencies (see above)

# Install Python package with build dependencies
pip install scikit-build-core pybind11
pip install -e .
```

## Troubleshooting

### carma not found

If you get an error about carma not being found:

```bash
# Check if carma is installed
ls /usr/local/include/carma

# If not found, install it (see instructions above)
```

### Armadillo not found

```bash
# Ubuntu/Debian
sudo apt-get install libarmadillo-dev

# macOS
brew install armadillo
```

### pybind11 not found

```bash
pip install pybind11
```

### CMake version too old

Update CMake:

```bash
# Ubuntu
sudo pip install cmake --upgrade

# macOS
brew upgrade cmake
```

## Verifying Installation

```python
from pilot import PILOT, RaFFLE
import numpy as np

# Test RaFFLE
X = np.random.randn(100, 5)
y = X[:, 0] + np.random.randn(100) * 0.1

model = RaFFLE(n_estimators=10, max_depth=3, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

print("Installation successful!")
print(f"R² score: {1 - np.mean((y - predictions)**2) / np.var(y):.3f}")
```

## Development Installation

For development with the scripts and benchmarks:

```bash
git clone https://github.com/yourusername/fast-model-trees.git
cd fast-model-trees

# Install with development dependencies
pip install -e ".[dev,benchmarks]"
```

## Getting Help

- **Issues**: https://github.com/yourusername/fast-model-trees/issues
- **Discussions**: https://github.com/yourusername/fast-model-trees/discussions
