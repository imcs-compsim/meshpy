# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
"""Test node utils of MeshPy."""

from meshpy.core.node import Node
from meshpy.utils.nodes import is_node_on_plane


def test_is_node_on_plane():
    """Test if node on plane function works properly."""

    # node on plane with origin_distance
    node = Node([1.0, 1.0, 1.0])
    assert is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=1.0)

    # node on plane with point_on_plane
    node = Node([1.0, 1.0, 1.0])
    assert is_node_on_plane(
        node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
    )

    # node not on plane with origin_distance
    node = Node([13.5, 14.5, 15.5])
    assert not is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=5.0)

    # node not on plane with point_on_plane
    node = Node([13.5, 14.5, 15.5])
    assert not is_node_on_plane(
        node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
    )
