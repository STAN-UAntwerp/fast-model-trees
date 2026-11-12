"""Sphinx configuration for the fast-model-trees documentation."""
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import the package from the source tree: the docs are built without compiling the C++ extension.
sys.path.insert(0, str(ROOT))

with open(ROOT / "pyproject.toml", "rb") as f:
    _pyproject = tomllib.load(f)

project = "fast-model-trees"
author = "Thomas Servotte, Jakob Raymaekers, Ruicong Yao"
copyright = "2024-2026, the fast-model-trees authors"
# On Read the Docs, show the version being built (e.g. "latest", "stable" or a tag).
release = os.environ.get("READTHEDOCS_VERSION_NAME", _pyproject["project"]["version"])
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# The compiled extension (pilot.cpilot) needs Armadillo/BLAS; mock it so autodoc can import pilot.
autodoc_mock_imports = ["pilot.cpilot"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "fast-model-trees"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/STAN-UAntwerp/fast-model-trees",
    "source_branch": "main",
    "source_directory": "docs/",
}
