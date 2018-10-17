# -*- coding: utf-8 -*-
"""
This script is used to compile the cyton code.
python3 find_close_nodes_setup.py build_ext --inplace
"""

# Python packages.
import numpy as np
import sys

# Cython stuff.
from distutils.core import setup
from Cython.Build import cythonize

# Check if the script is called with arguments.
if len(sys.argv) == 1:
    # No arguments are given, set them.
    sys.argv = [__file__, 'build_ext', '--inplace']

# Compile cython module.
setup(
    ext_modules=cythonize('find_close_nodes.pyx', annotate=True,
        build_dir='cython_build'),
    include_dirs=[np.get_include()]
    )
