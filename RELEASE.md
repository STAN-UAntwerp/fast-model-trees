# Release Process

This document describes how to publish a new version of `fast-model-trees` to PyPI.

## Prerequisites

1. Ensure you have set up the `PYPI_TOKEN` secret in your GitHub repository:
   - Go to https://github.com/STAN-UAntwerp/fast-model-trees/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_TOKEN`
   - Value: Your PyPI API token (get it from https://pypi.org/manage/account/token/)

## Publishing a New Release

The package is automatically published to PyPI when you push a version tag. Here's the workflow:

### 1. Update the version (optional)

The GitHub Action will automatically set the version based on the tag, but you may want to update it locally for consistency:

```bash
# Edit pyproject.toml and update the version field
# version = "0.1.2"  # for example
```

### 2. Commit your changes

```bash
git add .
git commit -m "Release v0.1.2"
```

### 3. Create and push a version tag

```bash
# Create a tag (format: v*.*.*)
git tag v0.1.2

# Push the tag to the fast-model-trees remote
git push fast-model-trees v0.1.2
```

### 4. Monitor the build

The GitHub Action will automatically:
- Build the package with the version from the tag
- Install all system dependencies (Armadillo, BLAS, LAPACK, carma)
- Compile the C++ extension
- Upload to PyPI

You can monitor the progress at:
https://github.com/STAN-UAntwerp/fast-model-trees/actions

### 5. Verify the release

Once the action completes, verify the new version is available:

```bash
pip install --upgrade fast-model-trees
python -c "import pilot; print(pilot.__version__)"
```

## Version Naming Convention

Follow semantic versioning (https://semver.org/):
- `v0.1.0` - Initial release
- `v0.1.1` - Patch release (bug fixes)
- `v0.2.0` - Minor release (new features, backward compatible)
- `v1.0.0` - Major release (breaking changes)

## Troubleshooting

### Build fails in GitHub Action

- Check the Actions tab for detailed logs
- Ensure all C++ dependencies are correctly specified
- Test the build locally first: `python -m build`

### Upload fails (403 Forbidden)

- Verify the `PYPI_TOKEN` secret is correctly set
- Ensure the token has upload permissions
- Check that the version number doesn't already exist on PyPI

### Wrong version published

- You cannot re-upload the same version to PyPI
- Delete the tag: `git tag -d v0.1.2 && git push fast-model-trees :refs/tags/v0.1.2`
- Create a new tag with the correct version: `git tag v0.1.3 && git push fast-model-trees v0.1.3`
