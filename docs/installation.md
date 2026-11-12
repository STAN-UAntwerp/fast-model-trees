# Installation

## From PyPI

```bash
pip install fast-model-trees
```

Pre-built wheels are available for Python 3.10 and newer on

| Platform | Architecture |
|----------|--------------|
| Linux (manylinux) | x86_64 |
| macOS 15 or newer | arm64 (Apple silicon) |
| Windows | AMD64 |

The wheels bundle the native libraries (Armadillo, BLAS, LAPACK), so nothing else needs
to be installed. The package is imported as `pilot`:

```python
from pilot import PILOT, RaFFLE
```

On other platforms pip falls back to building from the source distribution, which needs
the system dependencies listed below.

## From source

Building the C++ extension requires

- a C++17 compiler (GCC, Clang or MSVC)
- CMake 3.12 or newer
- [Armadillo](https://arma.sourceforge.net/) with BLAS and LAPACK

[carma](https://github.com/RUrlus/carma) (the Armadillo/NumPy bridge) and
[pybind11](https://pybind11.readthedocs.io/) are fetched automatically when they are
not installed.

**Ubuntu / Debian**

```bash
sudo apt-get install cmake g++ libopenblas-dev liblapack-dev libarmadillo-dev
```

**macOS**

```bash
brew install cmake armadillo
```

**Windows** (with [vcpkg](https://vcpkg.io/))

```bash
vcpkg install armadillo:x64-windows
```

`CMakeLists.txt` picks up the vcpkg toolchain from the `VCPKG_INSTALLATION_ROOT`
environment variable.

Then clone the repository and install it:

```bash
git clone https://github.com/STAN-UAntwerp/fast-model-trees.git
cd fast-model-trees
pip install -e ".[dev]"
```

## Verifying the installation

```python
import numpy as np
from pilot import RaFFLE

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 5))
y = X[:, 0] + 0.1 * rng.standard_normal(200)

model = RaFFLE(n_estimators=10, max_depth=3, oob_score=True).fit(X, y)
print(f"OOB R²: {model.oob_score_:.3f}")
```
