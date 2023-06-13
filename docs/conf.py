import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "insar-eventnet"
copyright = "2023, TODO"
author = "TODO"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
