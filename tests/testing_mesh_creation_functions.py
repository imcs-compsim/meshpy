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
This script is used to test the mesh creation functions.
"""

# Python imports.
import unittest
import numpy as np
import autograd.numpy as npAD
import os
import splinepy

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
from meshpy.utility import get_nodal_coordinates

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
    create_beam_mesh_helix,
    create_beam_flow_diverter,
)
from meshpy.mesh_creation_functions.beam_nurbs import (
    get_nurbs_curve_function_and_jacobian_for_integration,
)

# Testing imports.
from utils import testing_input, compare_test_result, compare_strings


def create_helix_function(
    radius, incline, *, transformation_factor=None, number_of_turns=None
):
    """Create and return a parametric function that represents a helix shape.
    The parameter coordinate can optionally be stretched to make the curve
    arc-length along the parameter coordinated non-constant and create a more
    complex curve for testing purposes.

    Args
    ----
    radius: float
        Radius of the helix
    incline: float
        Incline of the helix
    transformation_factor: float
        Factor to control the coordinate stretching (no direct physical interpretation)
    number_of_turns: float
        Number of turns the helix will have to get approximate boundaries for the transformation.
        This is only used for the transformation, not the actual geometry, as we return the
        function to create the geometry and not the geometry itself.
    """

    if transformation_factor is None and number_of_turns is None:
        # Identity transformation
        def transformation(t):
            return 1.0

    elif transformation_factor is not None and number_of_turns is not None:
        # Transform the parameter coordinate to make the function more complex
        def transformation(t):
            return (
                npAD.exp(transformation_factor * t / (2.0 * np.pi * number_of_turns))
                * t
                / npAD.exp(transformation_factor)
            )

    else:
        raise ValueError(
            "You have to set none or both optional parameters: "
            "transformation_factor and number_of_turns"
        )

    def helix(t):
        return npAD.array(
            [
                radius * npAD.cos(transformation(t)),
                radius * npAD.sin(transformation(t)),
                transformation(t) * incline / (2 * np.pi),
            ]
        )

    return helix


def create_testing_nurbs_curve():
    """Create a NURBS curve used for testing"""

    return splinepy.NURBS(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
        control_points=[[0, 0, 0], [1, 2, -1], [2, 0, 0]],
        weights=[[1.0], [1.0], [1.0]],
    )


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

        # Create beam elements.
        curve = create_testing_nurbs_curve()
        mat = MaterialReissner(radius=0.05)
        mesh = Mesh()
        _, length = create_beam_mesh_from_nurbs(
            mesh, Beam3rHerm2Line3, mat, curve, n_el=3, output_length=True
        )
        self.assertAlmostEqual(3.140204411551537, length, delta=mpy.eps_pos)

        # Check the output.
        input_file = InputFile()
        input_file.add(mesh)
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_nurbs_unit(self):
        """Unittest the function and jacobian creation in the create_beam_mesh_from_nurbs function"""

        curve = create_testing_nurbs_curve()
        r, dr, _, _ = get_nurbs_curve_function_and_jacobian_for_integration(
            curve, tol=10
        )

        t_values = [5.0 / 7.0, -0.3, 1.2]
        results_r = [
            [1.4285714285714286, 0.8163265306122449, -0.4081632653061225],
            [-0.6, -1.2, 0.6],
            [2.4, -0.8, 0.4],
        ]
        results_dr = [
            [2.0, -1.7142857142857144, 0.8571428571428572],
            [2.0, 4.0, -2.0],
            [2.0, -4.0, 2.0],
        ]

        for t, result_r, result_dr in zip(t_values, results_r, results_dr):
            self.assertTrue(np.allclose(r(t), result_r, atol=mpy.eps_pos))
            self.assertTrue(np.allclose(dr(t), result_dr, atol=mpy.eps_pos))

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

        # Get the helix curve function
        R = 2.0
        tz = 4.0  # incline
        n = 0.5  # number of turns
        helix = create_helix_function(
            R, tz, transformation_factor=2.0, number_of_turns=n
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

    def test_mesh_creation_functions_curve_3d_helix(self):
        """
        Create a helix from a parametric curve where the parameter is
        transformed so the arc length along the beam is not proportional to
        the parameter.
        """

        # Create input file.
        input_file = InputFile()

        # Add material and functions.
        mat = MaterialReissner()

        # Get the helix curve function
        R = 2.0
        tz = 4.0  # incline
        n = 1  # number of turns
        n_el = 5
        helix = create_helix_function(
            R, tz, transformation_factor=2.0, number_of_turns=n
        )

        helix_set = create_beam_mesh_curve(
            input_file, Beam3rHerm2Line3, mat, helix, [0.0, 2.0 * np.pi * n], n_el=n_el
        )
        input_file.add(helix_set)

        # Compare the coordinates with the ones from Mathematica.
        coordinates_mathematica = np.loadtxt(
            os.path.join(
                testing_input,
                "test_mesh_creation_functions_curve_3d_helix_mathematica.csv",
            ),
            delimiter=",",
        )
        self.assertLess(
            np.linalg.norm(
                coordinates_mathematica - get_nodal_coordinates(input_file.nodes)
            ),
            mpy.eps_pos,
        )

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_curve_3d_helix_length(self):
        """Create a helix from a parametric curve where and check that the
        correct length is returned.
        """

        input_file_1 = InputFile()
        input_file_2 = InputFile()
        mat = MaterialReissner()

        # Get the helix curve function
        R = 2.0
        tz = 4.0  # incline
        n = 1  # number of turns
        n_el = 3
        helix = create_helix_function(
            R, tz, transformation_factor=2.0, number_of_turns=n
        )

        args = [Beam3rHerm2Line3, mat, helix, [0.0, 2.0 * np.pi * n]]
        kwargs = {"n_el": n_el}

        helix_set_1 = create_beam_mesh_curve(input_file_1, *args, **kwargs)
        input_file_1.add(helix_set_1)

        helix_set_2, length = create_beam_mesh_curve(
            input_file_2, *args, output_length=True, **kwargs
        )
        input_file_2.add(helix_set_2)

        # Check the computed length
        self.assertAlmostEqual(length, 13.18763323790246, delta=1e-12)

        # Check that both meshes are equal
        compare_strings(
            self,
            input_file_1.get_string(header=False),
            input_file_2.get_string(header=False),
        )

    def test_mesh_creation_functions_curve_2d_sin(self):
        """Create a sin from a parametric curve."""

        # Create input file.
        input_file = InputFile()

        # Add material and functions.
        mat = MaterialReissner()

        # Set parameters for the sin.
        n_el = 8

        # Create a helix with a parametric curve.
        def sin(t):
            return npAD.array([t, npAD.sin(t)])

        sin_set = create_beam_mesh_curve(
            input_file, Beam3rHerm2Line3, mat, sin, [0.0, 2.0 * np.pi], n_el=n_el
        )
        input_file.add(sin_set)

        # Compare the coordinates with the ones from Mathematica.
        coordinates_mathematica = np.loadtxt(
            os.path.join(
                testing_input,
                "test_mesh_creation_functions_curve_2d_sin_mathematica.csv",
            ),
            delimiter=",",
        )
        self.assertLess(
            np.linalg.norm(
                coordinates_mathematica - get_nodal_coordinates(input_file.nodes)
            ),
            mpy.eps_pos,
        )

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_curve_3d_curve_rotation(self):
        """Create a line from a parametric curve and prescribe the rotation."""

        # AD.
        from autograd import jacobian

        # Create input file.
        input_file = InputFile()

        # Add material and functions.
        mat = MaterialReissner()

        # Set parameters for the line.
        L = 1.1
        n_el = 4

        def curve(t):
            return npAD.array([L * t, t * t * L * L, 0.0])

        def rotation(t):
            rp2 = jacobian(curve)(t)
            rp = [rp2[0], rp2[1], 0]
            R1 = Rotation([1, 0, 0], t * 2 * np.pi)
            R2 = Rotation.from_basis(rp, [0, 0, 1])
            return R2 * R1

        sin_set = create_beam_mesh_curve(
            input_file,
            Beam3rHerm2Line3,
            mat,
            curve,
            [0.0, 1.0],
            n_el=n_el,
            function_rotation=rotation,
        )
        input_file.add(sin_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_curve_3d_line(self):
        """
        Create a line from a parametric curve. Once the interval is in
        ascending order, once in descending. This tests checks that the
        elements are created with the correct tangent vectors.
        """

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)

        # Create a line with a parametric curve (and a transformed parameter).
        def line(t):
            factor = 2
            t_trans = npAD.exp(factor * t / (2.0 * np.pi)) * t / npAD.exp(factor)
            return npAD.array([t_trans, 0, 0])

        # Create mesh.
        set_1 = create_beam_mesh_curve(
            input_file, Beam3rHerm2Line3, mat, line, [0.0, 5.0], n_el=3
        )
        input_file.translate([0, 1, 0])
        set_2 = create_beam_mesh_curve(
            input_file, Beam3rHerm2Line3, mat, line, [5.0, 0.0], n_el=3
        )
        input_file.add(set_1, set_2)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_helix_no_rotation(self):
        """Create a helix and compare it with the reference file."""

        ## Helix angle and height helix combination
        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.5, shear_correction=1.0)

        # Add simple line to verify that the helix creation does not alter additional meshes
        create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
        )

        # Create helix.
        helix_set = create_beam_mesh_helix(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            helix_angle=np.pi / 4,
            height_helix=10.0,
            l_el=5.0,
        )
        input_file.add(helix_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

        ## Helix angle and turns
        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.5, shear_correction=1.0)

        # Add simple line to verify that the helix creation does not alter additional meshes
        create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
        )

        # Create helix.
        helix_set = create_beam_mesh_helix(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            helix_angle=np.pi / 4,
            turns=2.5 / np.pi,
            l_el=5.0,
        )
        input_file.add(helix_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

        ## Height helix and turns
        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.5, shear_correction=1.0)

        # Add simple line to verify that the helix creation does not alter additional meshes
        create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
        )

        # Create helix.
        helix_set = create_beam_mesh_helix(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            height_helix=10.0,
            turns=2.5 / np.pi,
            l_el=5.0,
        )
        input_file.add(helix_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_helix_rotation_offset(self):
        """Create a helix and compare it with the reference file."""

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.5, shear_correction=1.0)

        # Add simple line to verify that the helix creation does not alter additional meshes
        create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
        )

        # Create helix.
        helix_set = create_beam_mesh_helix(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [3.0, 0.0, 0.0],
            helix_angle=np.pi / 6,
            height_helix=10.0,
            l_el=5.0,
        )
        input_file.add(helix_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_helix_radius_zero(self):
        """Create a helix and compare it with the reference file."""

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.5, shear_correction=1.0)

        # Add simple line to verify that the helix creation does not alter additional meshes
        create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
        )

        # Create helix.
        helix_set = create_beam_mesh_helix(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            helix_angle=np.pi / 6,
            height_helix=80.0,
            n_el=4,
            warning_straight_line=False,
        )
        input_file.add(helix_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_helix_helix_angle_right_angle(self):
        """Create a helix and compare it with the reference file."""

        # Create input file.
        input_file = InputFile()

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.5, shear_correction=1.0)

        # Add simple line to verify that the helix creation does not alter additional meshes
        create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
        )

        # Create helix.
        helix_set = create_beam_mesh_helix(
            input_file,
            Beam3rHerm2Line3,
            mat,
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [2.0, 2.0, 1.0],
            helix_angle=np.pi / 2,
            height_helix=10.0,
            l_el=5.0,
            warning_straight_line=False,
        )
        input_file.add(helix_set)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_flow_diverter(self):
        """create a simple flow diverter based on beams"""

        # middle points may overlap
        mpy.check_overlapping_elements = False

        # Create input file, mesh and material
        input_file = InputFile()
        mesh = Mesh()
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.01, shear_correction=1.0)

        # create the flow diverter
        create_beam_flow_diverter(mesh, Beam3rHerm2Line3, mat, length=2, radius=1)

        # write output
        input_file.add(mesh)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_mesh_creation_functions_flow_diverter_interwooven(self):

        # middle points may overlap
        mpy.check_overlapping_elements = False

        # Create input file, mesh and material
        input_file = InputFile()
        mesh = Mesh()
        mat = MaterialReissner(youngs_modulus=1e5, radius=0.01, shear_correction=1.0)

        # create a flow diverter
        create_beam_flow_diverter(
            mesh,
            Beam3rHerm2Line3,
            mat,
            length=2,
            radius=1,
            n_turns=1,
            n_wire=2,
            n_el=2 * 2 * 1,
            interwooven=True,
        )

        # write output
        input_file.add(mesh)

        mesh.display_pyvista()

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
