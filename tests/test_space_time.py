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
"""Test the MeshPy beam to space-time surface functionality."""

import numpy as np
import pytest

from meshpy.core.conf import mpy
from meshpy.core.mesh import Mesh
from meshpy.four_c.element_beam import Beam3rHerm2Line3, Beam3rLine2Line2
from meshpy.four_c.input_file import InputFile
from meshpy.four_c.material import MaterialReissner
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_line,
)
from meshpy.space_time.beam_to_space_time import beam_to_space_time, mesh_to_data_arrays
from tests.test_performance import PerformanceTest


def get_name(beam_object):
    """Return the identifier for the given beam object."""
    if beam_object == Beam3rLine2Line2:
        return "linear"
    elif beam_object == Beam3rHerm2Line3:
        return "quadratic"
    else:
        raise TypeError("Got unexpected beam element")


@pytest.mark.parametrize("beam_type", [Beam3rLine2Line2, Beam3rHerm2Line3])
def test_space_time_straight(
    beam_type, assert_results_equal, get_corresponding_reference_file_path
):
    """Create the straight beam for the tests."""

    # Create the beam mesh in space
    beam_radius = 0.05
    mat = MaterialReissner(radius=beam_radius)
    mesh = Mesh()
    create_beam_mesh_line(mesh, beam_type, mat, [0, 0, 0], [6, 0, 0], n_el=3)

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 5, time_start=2.5)

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the dat file
    space_time_input_file = InputFile()
    space_time_input_file.add(space_time_mesh)
    additional_identifier = get_name(beam_type)
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        space_time_input_file,
    )

    # Check the mesh data arrays
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.parametrize("beam_type", [Beam3rLine2Line2, Beam3rHerm2Line3])
def test_space_time_curved(
    beam_type, assert_results_equal, get_corresponding_reference_file_path
):
    """Create a curved beam for the tests."""

    # Create the beam mesh in space
    beam_radius = 0.05
    mat = MaterialReissner(radius=beam_radius)
    mesh = Mesh()
    create_beam_mesh_arc_segment_2d(
        mesh, beam_type, mat, [0.5, 1, 0], 0.75, 0.0, np.pi * 2.0 / 3.0, n_el=3
    )

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 5)

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the dat file
    space_time_input_file = InputFile()
    space_time_input_file.add(space_time_mesh)
    additional_identifier = get_name(beam_type)
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        space_time_input_file,
        rtol=1e-12,
        atol=1e-12,
    )

    # Check the mesh data arrays
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.parametrize("beam_type", [Beam3rLine2Line2, Beam3rHerm2Line3])
@pytest.mark.parametrize("couple_nodes", [False, True])
def test_space_time_elbow(
    beam_type, couple_nodes, assert_results_equal, get_corresponding_reference_file_path
):
    """Create an elbow beam for the tests."""

    # Create the beam mesh in space
    beam_radius = 0.05
    mat = MaterialReissner(radius=beam_radius)
    mesh = Mesh()
    create_beam_mesh_line(mesh, beam_type, mat, [0, 0, 0], [1, 0, 0], n_el=3)
    create_beam_mesh_line(mesh, beam_type, mat, [1, 0, 0], [1, 1, 0], n_el=2)

    # Create the couplings
    if couple_nodes:
        mesh.couple_nodes(
            reuse_matching_nodes=True, coupling_dof_type=mpy.coupling_dof.fix
        )

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 5, time_start=1.69)

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    additional_identifier = get_name(beam_type) + ("_coupling" if couple_nodes else "")
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.parametrize("beam_type", [Beam3rLine2Line2, Beam3rHerm2Line3])
@pytest.mark.parametrize("couple_nodes", [False, True])
@pytest.mark.parametrize("arc_length", [False, True])
def test_space_time_varying_material_length(
    beam_type,
    couple_nodes,
    arc_length,
    assert_results_equal,
    get_corresponding_reference_file_path,
):
    """Create an elbow beam for the tests."""

    def beam_mesh_in_space_generator(time):
        """Create the beam mesh in space generator."""
        beam_radius = 0.05
        mat = MaterialReissner(youngs_modulus=100, radius=beam_radius, density=1)
        pos_y = 0.25 * (time - 1.7)

        mesh_1 = Mesh()
        create_beam_mesh_line(
            mesh_1, beam_type, mat, [np.sin(time), 0, 0], [2, pos_y, 0], n_el=3
        )

        mesh_2 = Mesh()
        create_beam_mesh_line(mesh_2, beam_type, mat, [2, pos_y, 0], [2, 3, 0], n_el=2)

        if arc_length:
            for i_mesh, line_mesh in enumerate([mesh_1, mesh_2]):
                for i_node, node in enumerate(line_mesh.nodes):
                    # This is a dummy arc length here, simply to achieve float values that
                    # are not matching at the corner node.
                    node.arc_length = i_node / 4.0 + i_mesh

        mesh = Mesh()
        mesh.add(mesh_1, mesh_2)
        if couple_nodes:
            mesh.couple_nodes(coupling_dof_type=mpy.coupling_dof.fix)
        return mesh

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(
        beam_mesh_in_space_generator, 6.9, 5, time_start=1.69
    )

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    additional_identifier = (
        get_name(beam_type)
        + ("_coupling" if couple_nodes else "")
        + ("_arc_length" if arc_length else "")
    )
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.performance
def test_space_time_performance():
    """Test the performance of the space time creation."""

    # These are the expected test times that should not be exceeded
    expected_times = {
        "space_time_create_mesh_in_space": 0.01,
        "space_time_create_mesh_in_time": 6.0,
    }
    test_performance = PerformanceTest(expected_times)

    # Create the beam mesh in space
    beam_type = Beam3rHerm2Line3
    beam_radius = 0.05
    mat = MaterialReissner(radius=beam_radius)
    mesh = Mesh()
    test_performance.time_function(
        "space_time_create_mesh_in_space",
        create_beam_mesh_line,
        args=[mesh, beam_type, mat, [0, 0, 0], [1, 0, 0]],
        kwargs={"n_el": 100},
    )

    # Create the beam mesh in time
    test_performance.time_function(
        "space_time_create_mesh_in_time",
        beam_to_space_time,
        args=[mesh, 6.9, 1000],
        kwargs={"time_start": 1.69},
    )

    assert test_performance.failed_tests == 0
