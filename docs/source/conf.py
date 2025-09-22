# Configuration file for the Sphinx documentation builder.
# For a full list of options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Ajoute le dossier `src` dans sys.path pour que Sphinx puisse importer ton package.
# Ici on remonte deux fois : docs/source → docs → racine → src
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# Infos générales affichées dans la documentation.
project = "Spark Batch Trainer"
author = "Rostand Surel"
copyright = "2025, Rostand Surel"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# Extensions activées pour enrichir la doc.
extensions = [
    "sphinx.ext.autodoc",     # Génère la doc depuis les docstrings
    "sphinx.ext.napoleon",    # Support des docstrings style NumPy/Google
    "sphinx.ext.viewcode",    # Ajoute des liens vers le code source
    "myst_parser",            # Permet d'utiliser Markdown (.md) en plus du reST
]

# Dossiers contenant les templates Jinja2 personnalisés (rarement nécessaire).
templates_path = ["_templates"]

# Exclusions : fichiers ou dossiers ignorés par Sphinx.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# Choix du thème visuel pour la doc.
# Ici : "sphinx_rtd_theme" (style ReadTheDocs, plus moderne que "alabaster").
html_theme = "sphinx_rtd_theme"

# Dossier contenant des fichiers statiques (CSS/JS/images personnalisées).
html_static_path = ["_static"]

# -- Autodoc configuration ---------------------------------------------------
# Personnalisation de l’autodoc (optionnel mais utile).
autodoc_typehints = "description"   # Ajoute les hints de type dans la description
autoclass_content = "both"          # Inclut docstring de la classe + __init__
