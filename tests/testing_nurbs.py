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
This script is used to test the mesh creation functions for NURBS.
"""

# Python imports
import unittest
import numpy as np
import os

# Meshpy imports
from meshpy import (
    InputFile,
    MaterialStVenantKirchhoff,
    Rotation,
)

# Geometry functions
from meshpy.mesh_creation_functions import (
    add_geomdl_nurbs_to_mesh,
    create_nurbs_hollow_cylinder_segment_2d,
    create_nurbs_flat_plate_2d,
    create_nurbs_brick,
)

# Testing imports
from testing_utility import (
    testing_input,
    compare_strings,
)


class TestNurbsMeshCreationFunction(unittest.TestCase):
    """
    Test the Nurbs Mesh creation functions
    """

    def test_hollow_cylinder_segment_2d(self):
        """Test the creation of a two dimensional hollow cylinder segment"""

        # Create the surface of a quarter of a hollow cylinder
        surf_obj = create_nurbs_hollow_cylinder_segment_2d(
            1.74, 2.46, np.pi * 5 / 6, n_ele_u=2, n_ele_v=3
        )

        # Create input file
        input_file = InputFile()

        # Add material
        mat = MaterialStVenantKirchhoff(youngs_modulus=50, nu=0.19, density=5.3e-7)

        # Create patch set
        element_description = (
            "KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
        )

        patch_set = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set)

        # Compare with the reference file
        compare_strings(
            self,
            "test_meshpy_create_nurbs_hollow_cylinder_segment_2d",
            os.path.join(
                testing_input,
                "test_meshpy_create_nurbs_hollow_cylinder_segment_2d_reference.dat",
            ),
            input_file.get_string(header=False),
        )

    def test_create_nurbs_flat_plate_2d(self):
        """Test the creation of a two dimensional flat plate"""

        # Create the surface of a flat plate
        surf_obj = create_nurbs_flat_plate_2d(0.75, 0.91, n_ele_u=2, n_ele_v=5)

        # Create input file
        input_file = InputFile()

        # Add material
        mat = MaterialStVenantKirchhoff(youngs_modulus=710, nu=0.19, density=5.3e-7)

        # Create patch set
        element_description = (
            "KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
        )

        patch_set = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set)

        # Compare with the reference file
        compare_strings(
            self,
            "test_meshpy_create_nurbs_flat_plate_2d",
            os.path.join(
                testing_input, "test_meshpy_create_nurbs_flat_plate_2d_reference.dat"
            ),
            input_file.get_string(header=False),
        )

    def test_create_nurbs_brick(self):
        """Test the creation of a brick"""

        # Create the surface of a flat plate
        vol_obj = create_nurbs_brick(1.5, 3.0, 2.4, n_ele_u=2, n_ele_v=3, n_ele_w=4)

        # Create input file
        input_file = InputFile()

        # Add material
        mat = MaterialStVenantKirchhoff(youngs_modulus=710, nu=0.19, density=5.3e-7)

        # Create patch set
        element_description = "GP 3 3 3"

        patch_set = add_geomdl_nurbs_to_mesh(
            input_file,
            vol_obj,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set)

        # Compare with the reference file
        compare_strings(
            self,
            "test_meshpy_create_nurbs_brick",
            os.path.join(testing_input, "test_meshpy_create_nurbs_brick_reference.dat"),
            input_file.get_string(header=False),
        )

    def test_rotation_nurbs_surface_mesh(self):
        """Test the rotation of a NURBS mesh"""

        # Create the surface
        surf_obj = create_nurbs_hollow_cylinder_segment_2d(
            1.74, 2.46, np.pi * 3 / 4, n_ele_u=5, n_ele_v=2
        )

        # Create input file
        input_file = InputFile()

        # Create patch set
        mat = MaterialStVenantKirchhoff(youngs_modulus=650, nu=0.20, density=4.2e-7)

        element_description = (
            "KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
        )

        patch_set = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set)

        input_file.rotate(Rotation([1, 2, 3], np.pi * 7 / 6))

        # Compare with the reference file
        compare_strings(
            self,
            "test_meshpy_rotation_nurbs_surface_mesh",
            os.path.join(
                testing_input, "test_meshpy_rotation_nurbs_surface_mesh_reference.dat"
            ),
            input_file.get_string(header=False),
        )

    def test_translate_nurbs_surface_mesh(self):
        """Test the translation of a NURBS surface mesh"""

        # Create the surface
        surf_obj = create_nurbs_flat_plate_2d(0.87, 1.35, n_ele_u=2, n_ele_v=3)

        # Create input file
        input_file = InputFile()

        # Create patch set
        mat = MaterialStVenantKirchhoff(youngs_modulus=430, nu=0.10, density=4.2e-7)

        element_description = (
            "KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
        )

        patch_set = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set)

        input_file.translate([-1.6, -2.3, 3.7])

        # Compare with the reference file
        compare_strings(
            self,
            "test_meshpy_translate_nurbs_surface_mesh",
            os.path.join(
                testing_input, "test_meshpy_translate_nurbs_surface_mesh_reference.dat"
            ),
            input_file.get_string(header=False),
        )

    def test_couple_nurbs_meshes(self):
        """Test the coupling of NURBS surface meshes"""

        # Create input file
        input_file = InputFile()

        # Create the first surface object
        surf_obj_1 = create_nurbs_hollow_cylinder_segment_2d(
            0.65, 1.46, np.pi * 2 / 3, n_ele_u=3, n_ele_v=2
        )

        # Create first patch set
        mat = MaterialStVenantKirchhoff(youngs_modulus=430, nu=0.10, density=4.2e-7)

        element_description = (
            "KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
        )

        patch_set_1 = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj_1,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set_1)

        input_file.rotate(Rotation([0, 0, 1], np.pi / 3))

        # Create the second surface object
        surf_obj_2 = create_nurbs_hollow_cylinder_segment_2d(
            0.65, 1.46, np.pi / 3, n_ele_u=3, n_ele_v=2
        )

        patch_set_2 = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj_2,
            material=mat,
            element_description=element_description,
        )

        input_file.add(patch_set_2)

        input_file.couple_nodes(reuse_matching_nodes=True)

        # Compare with the reference file
        compare_strings(
            self,
            "test_meshpy_couple_nurbs_meshes",
            os.path.join(
                testing_input, "test_meshpy_couple_nurbs_meshes_reference.dat"
            ),
            input_file.get_string(header=False),
        )
