import sys
from pathlib import Path

import os
import sys
from pathlib import Path

from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../D4Xgui'))

# Mock streamlit to avoid secrets errors during import
sys.modules['streamlit'] = MagicMock()

# Add the root directory of the project to sys.path
root_dir = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_dir))

# Add 'pages' and 'tools' directories to sys.path
sys.path.insert(0, str(root_dir / 'pages'))
sys.path.insert(0, str(root_dir / 'tools'))

# Add 'docs' directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'D4Xgui'
copyright = '2025, Miguel Bernecker'
author = 'Miguel Bernecker'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autodoc_mock_imports = [
    "streamlit",
    "streamlit_extras",
    "pandas",
    "numpy",
    "plotly",
    "dill",
    "scipy",
    "PIL",
    'D47crunch',
    'ogls',
    'matplotlib'
]


extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
        'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    'titles_only': True,
    'navigation_depth': 3,
}


