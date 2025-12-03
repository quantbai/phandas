Installation
============

Install from PyPI
-----------------

The simplest way::

    pip install phandas

Install from Source
-------------------

For development::

    git clone https://github.com/quantbai/phandas.git
    cd phandas
    pip install -e .

Build documentation (optional)::

    pip install -r docs/requirements.txt
    cd docs
    make html

Requirements
------------

- Python 3.8+
- numpy >= 2.0.0
- pandas >= 2.0.0, < 3.0.0
- matplotlib >= 3.7.0
- ccxt >= 4.0.0
- scipy >= 1.9.0
- python-okx >= 0.4.0
- requests >= 2.25.0

Verify Installation
-------------------

::

    python -c "import phandas; print(phandas.__version__)"

If you see the version number, installation was successful.
