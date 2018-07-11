# -*- coding: utf-8 -*-
"""
This script is used to compile the cyton code.
python3 setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('find_close_nodes.pyx', annotate=True)
    )
