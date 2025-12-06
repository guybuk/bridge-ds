# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bridge-ds"
copyright = "2024, Guy Bukchin"
author = "Guy Bukchin"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
#
nbsphinx_prolog = """
{{ env.docname.split('/')[-1].replace('_', ' ').title() }}
==========================================================

`Download this notebook from GitHub <https://raw.githubusercontent.com/guybuk/bridge-ds/main/docs/source/{{env.docname}}.ipynb>`_

.. raw:: html

    <a target="_blank" href="https://colab.research.google.com/github/guybuk/bridge-ds/blob/main/docs/source/{{env.docname}}.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
|
"""

# nbsphinx_execute = "never"
