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
"""This script is used to test the functionality of the geometry sets."""

from typing import Callable

import pytest

from meshpy.core.conf import mpy
from meshpy.core.element_beam import Beam
from meshpy.core.geometry_set import GeometrySet, GeometrySetNodes
from meshpy.core.node import Node


@pytest.fixture()
def assert_geometry_set_add_operator() -> Callable:
    """Return a function to check the results in the geometry set operator
    tests."""

    def _compare_results(
        mesh_objects, combined_geometry, set_1_geometry, set_2_geometry
    ):
        """Compare the results."""

        # Check that the added geometry set contains the combined geometry
        assert len(combined_geometry) == 5
        assert combined_geometry[0] is mesh_objects[2]
        assert combined_geometry[1] is mesh_objects[3]
        assert combined_geometry[2] is mesh_objects[4]
        assert combined_geometry[3] is mesh_objects[0]
        assert combined_geometry[4] is mesh_objects[1]

        # Check that the original sets are not modified
        assert len(set_1_geometry) == 3
        assert set_1_geometry[0] is mesh_objects[0]
        assert set_1_geometry[1] is mesh_objects[1]
        assert set_1_geometry[2] is mesh_objects[2]
        assert len(set_2_geometry) == 3
        assert set_2_geometry[0] is mesh_objects[2]
        assert set_2_geometry[1] is mesh_objects[3]
        assert set_2_geometry[2] is mesh_objects[4]

    return _compare_results


@pytest.mark.parametrize(
    ("mesh_object", "mesh_object_args"),
    [
        (Node, [[1, 2, 3]]),
        (Beam, []),
    ],
)
def test_core_geometry_set_add_operator(
    mesh_object, mesh_object_args, assert_geometry_set_add_operator
):
    """Test that geometry sets can be added to each other.

    We test this once with a point geometry set based on nodes and once
    with a line geometry set based on beam elements.
    """

    mesh_objects = [mesh_object(*mesh_object_args) for _ in range(5)]
    set_1 = GeometrySet(mesh_objects[:3])
    set_2 = GeometrySet(mesh_objects[2:])
    combined_set = set_2 + set_1
    combined_geometry = combined_set.get_geometry_objects()

    assert_geometry_set_add_operator(
        mesh_objects,
        combined_geometry,
        set_1.get_geometry_objects(),
        set_2.get_geometry_objects(),
    )


@pytest.mark.parametrize("geometry_type", [mpy.geo.point, mpy.geo.line])
def test_core_geometry_set_nodes_add_operator(
    geometry_type, assert_geometry_set_add_operator
):
    """Test that node based geometry sets can be added to each other."""

    mesh_objects = [Node([1, 2, 3]) for _ in range(5)]
    set_1 = GeometrySetNodes(geometry_type, nodes=mesh_objects[:3])
    set_2 = GeometrySetNodes(geometry_type, nodes=mesh_objects[2:])
    combined_set = set_2 + set_1
    combined_geometry = combined_set.get_all_nodes()

    assert_geometry_set_add_operator(
        mesh_objects, combined_geometry, set_1.get_all_nodes(), set_2.get_all_nodes()
    )
