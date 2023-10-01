"""Sphinx configuration."""
project = "Clehrity"
author = "Ben Lengerich"
copyright = "2023, Ben Lengerich"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
