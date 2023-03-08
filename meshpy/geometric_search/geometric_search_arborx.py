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
This file defines the interface to the ArborX geometric search functionality.
"""

# Python modules
import sys
import os

# Set path so ArborX binary will be found
sys.path.append(os.path.dirname(__file__))

# Import the ArborX wrapper
try:
    import geometric_search_arborx_lib

    arborx_available = True
except:
    arborx_available = False


class KokkosScopeGuardWrapper:
    """
    Wrap the initialize and finalize calls to Kokkos.
    """

    def __init__(self):
        """Call initialize when this object is created."""
        geometric_search_arborx_lib.kokkos_initialize()

    def __del__(self):
        """
        Finalize Kokkos after this object goes out of scope,
        i.e., at the end of this modules lifetime.
        """
        geometric_search_arborx_lib.kokkos_finalize()


if arborx_available:
    # Create the scope guard
    kokkos_scope_guard_wrapper = KokkosScopeGuardWrapper()


def find_close_points_arborx(point_coordinates, tol):
    """Call the ArborX implementation of find close_points."""
    if arborx_available:
        n_dim = point_coordinates.shape[1]
        if not n_dim == 3:
            raise ValueError(
                "ArborX geometric search is currently only implemented for 3 dimensions"
            )
        return geometric_search_arborx_lib.find_close_points(point_coordinates, tol)
    else:
        raise ModuleNotFoundError("ArborX functionality is not available")
