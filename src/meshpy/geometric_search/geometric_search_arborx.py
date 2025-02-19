# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This file defines the interface to the ArborX geometric search
functionality."""

from meshpy.utils.environment import is_arborx_available

if is_arborx_available():
    from meshpy.geometric_search import geometric_search_arborx_lib  # type: ignore


class KokkosScopeGuardWrapper:
    """Wrap the initialize and finalize calls to Kokkos."""

    def __init__(self):
        """Call initialize when this object is created."""
        geometric_search_arborx_lib.kokkos_initialize()

    def __del__(self):
        """Finalize Kokkos after this object goes out of scope, i.e., at the
        end of this modules lifetime."""
        geometric_search_arborx_lib.kokkos_finalize()


if is_arborx_available():
    # Create the scope guard
    kokkos_scope_guard_wrapper = KokkosScopeGuardWrapper()


def find_close_points_arborx(point_coordinates, tol):
    """Call the ArborX implementation of find close_points."""
    if is_arborx_available():
        return geometric_search_arborx_lib.find_close_points(point_coordinates, tol)
    else:
        raise ModuleNotFoundError("ArborX functionality is not available")
