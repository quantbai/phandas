from setuptools import setup, find_packages

setup(
    name='phandas',
    version='0.0.1',
    author='Phantom Management',
    author_email='quantbai@gmail.com',
    description='A framework for multi-factor backtesting in cryptocurrency markets by Phantom Management.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/quantbai/phandas',  # 根據您的資訊更新
    packages=find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
)
