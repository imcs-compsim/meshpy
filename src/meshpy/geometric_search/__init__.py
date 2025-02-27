# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This module defines geometric search functionality.

This module contains functionality to find points close to each other in
a point cloud. Currently, three different implementations for the actual
search algorithm are available (depending on your installation/setup):

- `brute_force_cython`: A brute force algorithm implemented in Cython,
  scales with $\mathcal{O}(n^2)$, but for a small number of points
  ($n<200$) this is the fastest algorithm since compared to the others
  it does not have any setup costs.

- `kd_tree_scipy`: Uses a BVH implementation provided by scipy. This
  scales with $\mathcal{O}(n\ \log{n})$.

- `boundary_volume_hierarchy_arborx` : Uses a BVH implementation
  provided by ArborX. This also scales with $\mathcal{O}(n\ \log{n})$
  but due to a more optimised implementation is a few times faster
  than the scipy implementation.

The `find_close_points` function automatically chooses the fastest
(available) implementation for the given point array.

Consult the `README.md` regarding install and testing options for
different implementations.
"""
