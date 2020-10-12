# -*- coding: utf-8 -*-
"""
This script is used to test the mesh creation functions.
"""

# Python imports.
import unittest
import numpy as np
import os

# Meshpy imports.
from meshpy import (mpy, InputFile, MaterialReissner, Beam3rHerm2Line3,
    MaterialEulerBernoulli, Beam3eb, Rotation, BoundaryCondition)

# Geometry functions.
from meshpy.mesh_creation_functions import (create_beam_mesh_arc_segment,
    create_beam_mesh_arc_segment_2d, create_beam_mesh_stent,
    create_fibers_in_rectangle)

# Testing imports.
from tests.testing_utility import testing_input, compare_strings


class TestMeshCreationFunctions(unittest.TestCase):
    """
    Test the mesh creation functions.
    """

    def test_arc_segment(self):
        """Create a circular segment and compare it with the reference file."""

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and function.
        mat = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Create mesh.
        mesh = create_beam_mesh_arc_segment(input_file, Beam3rHerm2Line3, mat,
            [3, 6, 9.2], Rotation([4.5, 7, 10], np.pi / 5), 10, np.pi / 2.3,
            n_el=5)

        # Add boundary conditions.
        input_file.add(BoundaryCondition(mesh['start'],
            'rb', bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(mesh['end'],
            'rb', bc_type=mpy.bc.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_segment_reference.dat')
        compare_strings(self,
            'test_meshpy_segment',
            ref_file,
            input_file.get_string(header=False))

    def test_arc_segment_2d(self):
        """
        Create a circular segments in 2D.
        """

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and function.
        mat = MaterialReissner(radius=0.1)

        # Create mesh.
        mesh1 = create_beam_mesh_arc_segment_2d(input_file, Beam3rHerm2Line3,
            mat, [1.0, 2.0, 0.0], 1.5, np.pi * 0.25, np.pi * (1.0 + 1.0 / 3.0),
            n_el=5)
        mesh2 = create_beam_mesh_arc_segment_2d(input_file, Beam3rHerm2Line3,
            mat, [1.0, 2.0, 0.0] - 2.0 * 0.5 * np.array([1, np.sqrt(3), 0]),
            0.5, np.pi / 3.0, -np.pi, n_el=3, start_node=input_file.nodes[-1])

        # Add boundary conditions.
        input_file.add(BoundaryCondition(mesh1['start'],
            'rb1', bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(mesh1['end'],
            'rb2', bc_type=mpy.bc.neumann))
        input_file.add(BoundaryCondition(mesh2['start'],
            'rb3', bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(mesh2['end'],
            'rb4', bc_type=mpy.bc.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_segment_2d_reference.dat')
        compare_strings(self,
            'test_meshpy_segment',
            ref_file,
            input_file.get_string(header=False))

    def test_stent(self):
        """
        Test the stent creation function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and function.
        mat = MaterialReissner()

        # Create mesh.
        create_beam_mesh_stent(input_file, Beam3rHerm2Line3, mat, 0.11, 0.02,
            5, 8, fac_bottom=0.6, fac_neck=0.52, fac_radius=0.36,
            alpha=0.47 * np.pi, n_el=2)

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_mesh_stent_reference.dat')
        compare_strings(
            self,
            'test_mesh_stent',
            ref_file,
            input_file.get_string(header=False))

    def test_fibers_in_rectangle(self):
        """
        Test the create_fibers_in_rectangle function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Create mesh.
        mat = MaterialEulerBernoulli()
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 45, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 0, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 90, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, -90, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 235, 0.45, 0.35)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            1, 4, 30, 0.45, 5)
        input_file.translate([0, 0, 1])
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 30, 0.45, 0.9)
        input_file.translate([0, 0, 1])

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_mesh_fiber_rectangle_reference.dat')
        compare_strings(
            self,
            'test_mesh_fiber_rectangle',
            ref_file,
            input_file.get_string(header=False))

    def test_fibers_in_rectangle_offset(self):
        """
        Test the create_fibers_in_rectangle function with using the offset
        option.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Create mesh.
        mat = MaterialEulerBernoulli()
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 45, 0.45, 0.35)
        create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 45, 0.45, 0.35, offset=0.1)

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_mesh_fiber_rectangle_offset_reference.dat')
        compare_strings(
            self,
            'test_mesh_fiber_rectangle_offset',
            ref_file,
            input_file.get_string(header=False))

    def test_fibers_in_rectangle_return_set(self):
        """
        Test the set returned by the create_fibers_in_rectangle function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Create mesh.
        mat = MaterialEulerBernoulli()
        beam_set = create_fibers_in_rectangle(input_file, Beam3eb, mat,
            4, 1, 45, 0.45, 0.35)
        input_file.add(beam_set)

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_mesh_fiber_rectangle_return_sets_reference.dat')
        compare_strings(
            self,
            'test_mesh_fiber_return_sets_rectangle',
            ref_file,
            input_file.get_string(header=False))


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
