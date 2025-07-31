# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import rosoku
from pathlib import Path

page_repository_base = Path("~") / "git" / "rosoku.github.io"

project = "Rosoku"
copyright = "2025, Simon Kojima"
author = "Simon Kojima"
release = rosoku.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_multiversion",
    "numpydoc",
]

current_version = os.environ.get("SMV_VERSION", "")

if current_version == "latest" or current_version == "main":
    examples_are_enabled = True
else:
    examples_are_enabled = False

autosummary_generate = True

smv_tag_whitelist = r"^v\d+\.\d+.*$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"

# smv_rename_latest_version = True
# smv_branch_labels = {
#    "main": "latest",
# }

# import sphinx_rtd_theme

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "default_mode": "light",
    "logo": {"image_light": "logo-light.png", "image_dark": "logo-dark.png"},
    "version_dropdown": True,
    "version_info": {
        "name": "Version",
        "current_version": release,
    },
    "navbar_end": ["theme-switcher", "version-switcher"],
    "switcher": {
        "json_url": page_repository_base / "versions.json",
        "version_match": release,
    },
}
html_context = {
    "version_match": release,
}

if examples_are_enabled:
    extensions += ["sphinx_gallery.gen_gallery"]
    sphinx_gallery_conf = {
        "examples_dirs": "../../examples",
        "gallery_dirs": "auto_examples",
        "filename_pattern": r"example_",
    }
else:
    pass


source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
