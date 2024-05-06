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
"""
Test utilities of MeshPy.
"""

# Python imports
import unittest

# Meshpy imports
from meshpy.utility import is_node_on_plane

from meshpy.node import Node


class TestUtilities(unittest.TestCase):
    """Test utilities from the meshpy.utility module."""

    def test_is_node_on_plane(self):
        """Test if node on plane function works properly."""

        # node on plane with origin_distance
        node = Node([1.0, 1.0, 1.0])
        self.assertTrue(
            is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=1.0)
        )

        # node on plane with point_on_plane
        node = Node([1.0, 1.0, 1.0])
        self.assertTrue(
            is_node_on_plane(
                node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
            )
        )

        # node not on plane with origin_distance
        node = Node([13.5, 14.5, 15.5])
        self.assertFalse(
            is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=5.0)
        )

        # node not on plane with point_on_plane
        node = Node([13.5, 14.5, 15.5])
        self.assertFalse(
            is_node_on_plane(
                node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
            )
        )


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
