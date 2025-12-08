# Development Guide

This guide covers development setup, running benchmarks, and reproducing paper results.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python environment management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
uv sync
```

This creates a virtual environment and installs all required dependencies from `pyproject.toml`.

### 3. Build C++ Extension

For building the C++ extension from source, see [INSTALL.md](INSTALL.md) for detailed platform-specific instructions.

## Running Examples

Run the RaFFLE example:
```bash
uv run python scripts/raffle_example.py
```

## Feature Importance Testing

Test feature importance implementation:
```bash
uv run python scripts/test_feature_importance.py
```

## RaFFLE Benchmark

To reproduce the benchmark results from the RaFFLE paper:

### 1. Download Datasets

```bash
uv run python scripts/download_data.py
```

### 2. Run Benchmark

```bash
uv run python scripts/benchmark.py
```

Results will be stored in the `Output` folder.

### 3. Generate Paper Plots

Create all plots from the paper:
```bash
uv run python scripts/paperplots.py --all
```

## Repository Structure

```
├── pilot/              # Python package
│   ├── __init__.py
│   ├── c_ensemble.py   # RaFFLE and PILOTWrapper classes
│   └── constants.py    # Configuration constants
├── scripts/            # Development scripts
│   ├── benchmark.py
│   ├── paperplots.py
│   └── ...
├── *.cpp, *.h          # C++ source files
├── CMakeLists.txt      # CMake configuration
└── pyproject.toml      # Package configuration
```

## Contributing

When contributing:

1. Ensure all tests pass
2. Follow the existing code style
3. Update documentation as needed
4. Add tests for new features

## Release Process

See [RELEASE.md](RELEASE.md) for instructions on publishing new versions to PyPI.
