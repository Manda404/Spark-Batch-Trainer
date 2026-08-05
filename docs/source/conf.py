"""Sphinx configuration for Spark Batch Trainer."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "Spark Batch Trainer"
author = "Rostand Surel"
copyright = f"{date.today().year}, {author}"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {release}"
html_show_sourcelink = False

autodoc_typehints = "description"
autoclass_content = "class"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True
autosectionlabel_prefix_document = True

rst_prolog = f"""
.. |date| replace:: {date.today():%Y-%m-%d}
"""
