import sys
from pathlib import Path

# 讓 Sphinx 找到 phandas 源代碼
sys.path.insert(0, str(Path(__file__).parent.parent / "phandas"))

# -- Project information -----------------------------------------------------
project = 'phandas'
copyright = '2025, Phantom Management'
author = 'Phantom Management'
release = '0.14.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # 自動從 docstring 生成文檔
    'sphinx.ext.napoleon',     # 支持 NumPy/Google 格式
    'sphinx.ext.intersphinx',  # 連結到其他文檔
    'myst_parser',             # Markdown 支持
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_TW'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
