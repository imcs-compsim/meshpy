# -*- coding: utf-8 -*-
"""
This script is used to compile the cyton code.
python3 find_close_nodes_setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(
    ext_modules=cythonize('find_close_nodes.pyx', annotate=True),
    include_dirs=[np.get_include()]
    )
