import re
from pathlib import Path
from setuptools import setup, find_packages


def get_version():
    init_file = Path(__file__).parent / 'phandas' / '__init__.py'
    content = init_file.read_text(encoding='utf-8')
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.M)
    return match.group(1) if match else '0.0.0'


setup(
    name='phandas',
    version=get_version(),
    author='Phantom Management',
    author_email='quantbai@gmail.com',
    description='A multi-factor quantitative trading framework for cryptocurrency markets.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/quantbai/phandas',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.0.0',
        'pandas>=2.0.0,<3.0.0',
        'matplotlib>=3.7.0',
        'ccxt>=4.0.0',
        'scipy>=1.9.0',
        'python-okx>=0.4.0',
        'requests>=2.25.0',
        'mcp>=0.1.0',
        'rich>=13.0.0',
    ],
    entry_points={
        'console_scripts': [
            'phandas-mcp=phandas.mcp_server:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.10',
)
