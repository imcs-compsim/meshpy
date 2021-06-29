# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
