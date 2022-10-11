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
This script is used to test the mesh creation functions.
"""

# Python imports.
import unittest
import numpy as np
import os

# Meshpy imports.
from testing_context import meshpy
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

# Geometry functions.
from meshpy.mesh_creation_functions import (
    create_beam_mesh_arc_segment,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_stent,
    create_fibers_in_rectangle,
    create_wire_fibers,
    create_beam_mesh_from_nurbs,
)

# Testing imports.
from testing_utility import testing_input, compare_strings


class TestMeshCreationFunctions(unittest.TestCase):
    """
    Test the mesh creation functions.
    """

    def test_arc_segment(self):
        """Create a circular segment and compare it with the reference file."""

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

        # Add material and function.
        mat = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)

        # Create mesh.
        mesh = create_beam_mesh_arc_segment(
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
        ref_file = os.path.join(testing_input, "test_meshpy_segment_reference.dat")
        compare_strings(
            self, "test_meshpy_segment", ref_file, input_file.get_string(header=False)
        )

    def test_arc_segment_2d(self):
        """
        Create a circular segments in 2D.
        """

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

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
        ref_file = os.path.join(testing_input, "test_meshpy_segment_2d_reference.dat")
        compare_strings(
            self, "test_meshpy_segment", ref_file, input_file.get_string(header=False)
        )

    def test_stent(self):
        """
        Test the stent creation function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

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
        ref_file = os.path.join(testing_input, "test_mesh_stent_reference.dat")
        compare_strings(
            self, "test_mesh_stent", ref_file, input_file.get_string(header=False)
        )

    def test_fibers_in_rectangle(self):
        """
        Test the create_fibers_in_rectangle function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

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
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 1, 4, 30, 0.45, 5)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 30, 0.45, 0.9)
        input_file.translate([0, 0, 1])

        # Check the output.
        ref_file = os.path.join(
            testing_input, "test_mesh_fiber_rectangle_reference.dat"
        )
        compare_strings(
            self,
            "test_mesh_fiber_rectangle",
            ref_file,
            input_file.get_string(header=False),
        )

    def test_fibers_in_rectangle_offset(self):
        """
        Test the create_fibers_in_rectangle function with using the offset
        option.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

        # Create mesh.
        mat = MaterialEulerBernoulli()
        create_fibers_in_rectangle(input_file, Beam3eb, mat, 4, 1, 45, 0.45, 0.35)
        create_fibers_in_rectangle(
            input_file, Beam3eb, mat, 4, 1, 45, 0.45, 0.35, offset=0.1
        )

        # Check the output.
        ref_file = os.path.join(
            testing_input, "test_mesh_fiber_rectangle_offset_reference.dat"
        )
        compare_strings(
            self,
            "test_mesh_fiber_rectangle_offset",
            ref_file,
            input_file.get_string(header=False),
        )

    def test_fibers_in_rectangle_return_set(self):
        """
        Test the set returned by the create_fibers_in_rectangle function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

        # Create mesh.
        mat = MaterialEulerBernoulli()
        beam_set = create_fibers_in_rectangle(
            input_file, Beam3eb, mat, 4, 1, 45, 0.45, 0.35
        )
        input_file.add(beam_set)

        # Check the output.
        ref_file = os.path.join(
            testing_input, "test_mesh_fiber_rectangle_return_sets_reference.dat"
        )
        compare_strings(
            self,
            "test_mesh_fiber_return_sets_rectangle",
            ref_file,
            input_file.get_string(header=False),
        )

    def test_wire(self):
        """
        Test the create_wire_fibers function
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

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
        ref_file = os.path.join(testing_input, "test_mesh_wire_reference.dat")
        compare_strings(
            self, "test_mesh_wire", ref_file, input_file.get_string(header=False)
        )

    def test_nurbs(self):
        """
        Test the create_beam_mesh_from_nurbs function.
        """

        # Setup the nurbs curve.
        from geomdl import NURBS

        curve = NURBS.Curve()
        curve.degree = 2
        curve.ctrlpts = [[0, 0, 0], [1, 2, -1], [2, 0, 0]]
        curve.knotvector = [0, 0, 0, 1, 1, 1]

        # Create beam elements.
        mat = MaterialReissner(radius=0.05)
        mesh = Mesh()
        create_beam_mesh_from_nurbs(mesh, Beam3rHerm2Line3, mat, curve, n_el=3)

        # Check the output.
        input_file = InputFile()
        input_file.add(mesh)
        ref_file = os.path.join(testing_input, "test_mesh_nurbs_reference.dat")
        compare_strings(
            self, "test_mesh_nurbs", ref_file, input_file.get_string(header=False)
        )


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
