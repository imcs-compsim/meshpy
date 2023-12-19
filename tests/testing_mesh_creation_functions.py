# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2023
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
This script is used to test the mesh creation functions.
"""

# Python imports.
import unittest
import numpy as np
import autograd.numpy as npAD
import os

# Meshpy imports.
from meshpy import (
    mpy,
    Mesh,
    InputFile,
    MaterialReissner,
    Beam3rHerm2Line3,
    MaterialEulerBernoulli,
    Beam3eb,
    Rotation,
    BoundaryCondition,
)
from meshpy.node import NodeCosserat

# Geometry functions.
from meshpy.mesh_creation_functions.beam_generic import create_beam_mesh_function
from meshpy.mesh_creation_functions import (
    create_beam_mesh_line,
    create_beam_mesh_curve,
    create_beam_mesh_arc_segment_via_rotation,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_stent,
    create_beam_mesh_line_at_node,
    create_beam_mesh_arc_at_node,
    create_fibers_in_rectangle,
    create_wire_fibers,
    create_beam_mesh_from_nurbs,
)

# Testing imports.
from testing_utility import testing_input, compare_test_result


class TestMeshCreationFunctions(unittest.TestCase):
    """
    Test the mesh creation functions.
    """

    def test_mesh_creation_functions_arc_segment(self):
        """Create a circular segment and compare it with the reference file."""

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)

        # Create mesh.
        mesh = create_beam_mesh_arc_segment_via_rotation(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [3, 6, 9.2],
            Rotation([4.5, 7, 10], np.pi / 5),
            10,
            np.pi / 2.3,
            n_el=5,
        )

        # Add boundary conditions.
        input_file.add(BoundaryCondition(mesh["start"], "rb", bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(mesh["end"], "rb", bc_type=mpy.bc.neumann))

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_arc_segment_2d(self):
        """
        Create a circular segments in 2D.
        """

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(radius=0.1)

        # Create mesh.
        mesh1 = create_beam_mesh_arc_segment_2d(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [1.0, 2.0, 0.0],
            1.5,
            np.pi * 0.25,
            np.pi * (1.0 + 1.0 / 3.0),
            n_el=5,
        )
        mesh2 = create_beam_mesh_arc_segment_2d(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [1.0, 2.0, 0.0] - 2.0 * 0.5 * np.array([1, np.sqrt(3), 0]),
            0.5,
            np.pi / 3.0,
            -np.pi,
            n_el=3,
            start_node=input_file.nodes[-1],
        )

        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(mesh1["start"], "rb1", bc_type=mpy.bc.dirichlet)
        )
        input_file.add(BoundaryCondition(mesh1["end"], "rb2", bc_type=mpy.bc.neumann))
        input_file.add(
            BoundaryCondition(mesh2["start"], "rb3", bc_type=mpy.bc.dirichlet)
        )
        input_file.add(BoundaryCondition(mesh2["end"], "rb4", bc_type=mpy.bc.neumann))

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_stent(self):
        """
        Test the stent creation function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner()

        # Create mesh.
        create_beam_mesh_stent(
            input_file,
            Beam3rHerm2Line3,
            mat,
            0.11,
            0.02,
            5,
            8,
            fac_bottom=0.6,
            fac_neck=0.52,
            fac_radius=0.36,
            alpha=0.47 * np.pi,
            n_el=2,
        )

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False), rtol=1e-10)

    def test_mesh_creation_functions_fibers_in_rectangle(self):
        """
        Test the create_fibers_in_rectangle function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile()

        # Create mesh.
        mat = MaterialEulerBernoulli()
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 45, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 0, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 90, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, -90, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 235, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(
            input_file, Beam3eb, mat, 1, 4, 30, 0.45, 5, fiber_element_length_min=0.2
        )
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 30, 0.45, 0.9)
        input_file.translate([0, 0, 1])

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_fibers_in_rectangle_reference_point(self):
        """
        Test the create_fibers_in_rectangle function with using the reference_point
        option.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile()

        # Create mesh.
        mat = MaterialEulerBernoulli()
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 45, 0.45, 0.35)
        reference_point = 0.5 * np.array([4.0, 1.0]) + 0.1 * np.array(
            [-1.0, 1.0]
        ) / np.sqrt(2.0)
        create_fibers_in_rectangle(
            input_file,
            Beam3eb,
            mat,
            4,
            1,
            45,
            0.45,
            0.35,
            reference_point=reference_point,
        )

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_fibers_in_rectangle_return_set(self):
        """
        Test the set returned by the create_fibers_in_rectangle function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile()

        # Create mesh.
        mat = MaterialEulerBernoulli()
        beam_set = create_fibers_in_rectangle(
            input_file, Beam3eb, mat, 4, 1, 45, 0.45, 0.35
        )
        input_file.add(beam_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_wire(self):
        """
        Test the create_wire_fibers function
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile()

        # Create two wires with different parameters.
        mat = MaterialEulerBernoulli(radius=0.05)
        mesh_1 = Mesh()
        set_1 = create_wire_fibers(mesh_1, Beam3eb, mat, 3.0, layers=2, n_el=2)
        mesh_2 = Mesh()
        set_2 = create_wire_fibers(
            mesh_2, Beam3eb, mat, 3.0, layers=2, n_el=2, radius=0.1
        )
        mesh_2.translate([0.0, 1.5, 0.0])
        input_file.add(mesh_1, mesh_2, set_1, set_2)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_nurbs(self):
        """
        Test the create_beam_mesh_from_nurbs function.
        """

        # Setup the NURBS curve.
        from geomdl import NURBS
        from geomdl import utilities

        curve = NURBS.Curve()
        curve.degree = 2
        curve.ctrlpts = [[0, 0, 0], [1, 2, -1], [2, 0, 0]]
        curve.knotvector = utilities.generate_knot_vector(
            curve.degree, len(curve.ctrlpts)
        )

        # Create beam elements.
        mat = MaterialReissner(radius=0.05)
        mesh = Mesh()
        create_beam_mesh_from_nurbs(mesh, Beam3rHerm2Line3, mat, curve, n_el=3)

        # Check the output.
        input_file = InputFile()
        input_file.add(mesh)
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_node_continuation(self):
        """Test that the node continuation function work as expected."""

        mesh = Mesh()
        mat = MaterialReissner(radius=0.1)
        mesh.add(mat)

        start_node = NodeCosserat([1, 2, 3], Rotation([0, 0, 0], 0))
        mesh.add(start_node)
        beam_set = create_beam_mesh_line_at_node(
            mesh, Beam3rHerm2Line3, mat, start_node, 1.2, n_el=1
        )
        beam_set = create_beam_mesh_arc_at_node(
            mesh,
            Beam3rHerm2Line3,
            mat,
            beam_set["end"],
            [0, 1, 0],
            1.0,
            np.pi * 0.5,
            n_el=1,
        )
        beam_set = create_beam_mesh_arc_at_node(
            mesh,
            Beam3rHerm2Line3,
            mat,
            beam_set["end"],
            [0, 1, 0],
            1.0,
            -np.pi * 0.5,
            n_el=2,
        )
        beam_set = create_beam_mesh_arc_at_node(
            mesh,
            Beam3rHerm2Line3,
            mat,
            beam_set["end"],
            [0, 0, 1],
            1.0,
            -np.pi * 3.0 / 4.0,
            n_el=1,
        )
        create_beam_mesh_line_at_node(
            mesh, Beam3rHerm2Line3, mat, beam_set["end"], 2.3, n_el=3
        )

        # Check the geometry
        input_file = InputFile()
        input_file.add(mesh)
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_node_continuation_accumulated(self):
        """Test that the arc node continuation function can be applied multiple times in a row.
        This function can lead to accumulated errors in the rotations if not implemented
        carefully."""

        mesh = Mesh()
        mat = MaterialReissner(radius=0.1)
        mesh.add(mat)

        n_segments = 100

        rotation_ref = Rotation([1, 0, 0], 0.5 * np.pi) * Rotation(
            [0, 0, 1], 0.5 * np.pi
        )
        start_node = NodeCosserat([1, 2, 3], rotation_ref)
        mesh.add(start_node)
        beam_set = {"end": start_node}
        angle = np.pi
        angle_increment = angle / n_segments
        axis = [1, 0, 0]
        for i in range(n_segments):
            beam_set = create_beam_mesh_arc_at_node(
                mesh,
                Beam3rHerm2Line3,
                mat,
                beam_set["end"],
                axis,
                1.0,
                angle_increment,
                n_el=2,
            )

        rotation_expected = Rotation(axis, angle) * rotation_ref
        rotation_actual = beam_set["end"].get_points()[0].rotation
        self.assertTrue(rotation_actual == rotation_expected)

    def test_mesh_creation_functions_element_length_option(self):
        """Test that the element length can be specified in the beam creation functions"""

        input_file = InputFile()
        mat = MaterialReissner(radius=0.1)

        l_el = 1.5

        mesh_line = Mesh()
        create_beam_mesh_line(
            mesh_line,
            Beam3rHerm2Line3,
            mat,
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 6.0],
            l_el=l_el,
        )

        mesh_line_long = Mesh()
        create_beam_mesh_line(
            mesh_line_long,
            Beam3rHerm2Line3,
            mat,
            [1.0, 2.0, 2.0],
            [3.0, 4.0, 8.0],
            l_el=100,
        )

        mesh_arc = Mesh()
        create_beam_mesh_arc_segment_via_rotation(
            mesh_arc,
            Beam3rHerm2Line3,
            mat,
            [1.0, 2.0, 3.0],
            Rotation([1, 3, 4], np.pi / 3.0),
            2.0,
            np.pi * 2.0 / 3.0,
            l_el=l_el,
        )

        # Set parameters for the helix.
        R = 2.0
        tz = 4.0  # incline
        n = 0.5  # number of turns

        def helix(t):
            factor = 2
            t_trans = npAD.exp(factor * t / (2.0 * np.pi * n)) * t / npAD.exp(factor)
            return npAD.array(
                [
                    R * npAD.cos(t_trans),
                    R * npAD.sin(t_trans),
                    t_trans * tz / (2 * np.pi),
                ]
            )

        mesh_curve = Mesh()
        create_beam_mesh_curve(
            mesh_curve,
            Beam3rHerm2Line3,
            mat,
            helix,
            [0.0, 2.0 * np.pi * n],
            l_el=l_el,
        )

        # Check the output
        input_file.add(mesh_line, mesh_line_long, mesh_arc, mesh_curve)
        compare_test_result(self, input_file.get_string(header=False), rtol=1e-10)

        # Check error messages for input parameters
        with self.assertRaises(ValueError):
            mesh = Mesh()
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                mat,
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 6.0],
                n_el=1,
                l_el=l_el,
            )
        with self.assertRaises(ValueError):
            mesh = Mesh()
            return create_beam_mesh_function(mesh, interval=[0.0, 1.0], l_el=2.0)


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
