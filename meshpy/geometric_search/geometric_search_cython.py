# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
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
"""This file defines the interface to the Cython geometric search
functionality."""

# Import python module
import os
import sys
import warnings

# Set path so the Cython binary will be found
sys.path.append(os.path.dirname(__file__))

# Import the Cython module
try:
    import geometric_search_cython_lib

    cython_available = True
except:
    cython_available = False


def find_close_points_brute_force_cython(
    point_coordinates, tol, *, n_points_performance_warning=5000
):
    """Call the Cython brute force implementation of find close_points."""
    if cython_available:
        n_points = len(point_coordinates)
        if n_points > n_points_performance_warning:
            warnings.warn(
                "The function find_close_points is called with the brute force algorithm "
                + f"with {n_points} points, for performance reasons other algorithms should be used!"
            )
        return geometric_search_cython_lib.find_close_points(point_coordinates, tol)
    else:
        raise ModuleNotFoundError(
            "Cython geometric search functionality is not available"
        )
