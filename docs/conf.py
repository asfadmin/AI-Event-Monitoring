project = "insar-eventnet"
copyright = "2023, TODO"
author = "TODO"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link"
]

nbsphinx_execute = 'never'

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
