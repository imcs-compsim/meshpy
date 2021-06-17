# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This script is used to compile the cyton code.
python3 find_close_points_setup.py build_ext --inplace
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
    ext_modules=cythonize('find_close_points.pyx', annotate=True,
        build_dir='cython_build'),
    include_dirs=[np.get_include()]
    )
