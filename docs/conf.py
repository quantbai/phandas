import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

project = 'phandas'
copyright = '2025, Phantom Management'
author = 'Phantom Management'
release = '0.16.0'

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
