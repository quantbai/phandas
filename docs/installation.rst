安裝
====

從 PyPI 安裝
-------------

最簡單的方式::

    pip install phandas

從源代碼安裝
-------------

用於開發::

    git clone https://github.com/quantbai/phandas.git
    cd phandas
    pip install -e .

構建文檔（可選）::

    pip install -r docs/requirements.txt
    cd docs
    make html

系統要求
--------

- Python 3.8+
- numpy >= 2.0.0
- pandas >= 2.0.0
- ccxt >= 4.0.0
- scipy >= 1.9.0

驗證安裝
--------

::

    python -c "import phandas; print(phandas.__version__)"

如果看到版本號，表示安裝成功。

