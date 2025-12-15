import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phandas import __version__

project = 'phandas'
copyright = '2025, Phantom Management'
author = 'Phantom Management'
release = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
