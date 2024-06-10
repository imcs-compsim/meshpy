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
    MaterialString,
    Rotation,
)

# Geometry functions
from meshpy.mesh_creation_functions import (
    add_geomdl_nurbs_to_mesh,
    create_nurbs_hollow_cylinder_segment_2d,
    create_nurbs_flat_plate_2d,
    create_nurbs_brick,
    create_nurbs_sphere_surface,
    create_nurbs_hemisphere_surface,
)

# Testing imports
from utils import testing_input, compare_test_result


class TestNurbsMeshCreationFunction(unittest.TestCase):
    """
    Test the Nurbs Mesh creation functions
    """

    def test_nurbs_hollow_cylinder_segment_2d(self):
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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_flat_plate_2d(self):
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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_brick(self):
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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_rotation_nurbs_surface(self):
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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_translate_nurbs_surface(self):
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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_couple_nurbs_meshes(self):
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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_sphere_surface(self):
        """Test the creating of the base patch of the surface of a sphere."""

        # Create input file
        input_file = InputFile()

        # Create the base of a sphere
        surf_obj = create_nurbs_sphere_surface(1, n_ele_u=3, n_ele_v=2)

        # Create first patch set
        mat = MaterialStVenantKirchhoff()

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
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_string_types(self):
        """Test the creating of a NURBS with strings for the element and material definition."""

        # Create input file
        input_file = InputFile()

        # Create the base of a sphere
        surf_obj = create_nurbs_flat_plate_2d(1, 3, n_ele_u=3, n_ele_v=2)

        # Create first patch set
        mat = MaterialString("STRING_MATERIAL")

        element_description = ()

        patch_set = add_geomdl_nurbs_to_mesh(
            input_file,
            surf_obj,
            material=mat,
            element_string="STRING_TYPE",
            element_description="STRING_DESCRIPTION",
        )

        input_file.add(patch_set)

        # Compare with the reference file
        compare_test_result(self, input_file.get_string(header=False))

    def test_nurbs_hemisphere_surface(self):
        """Test the creation of the surface of a hemisphere."""

        # Create input file
        input_file = InputFile()

        # Create the base of a sphere
        surfs = create_nurbs_hemisphere_surface(2.5, n_ele_uv=2)

        # Create first patch set
        mat = MaterialStVenantKirchhoff()

        element_description = (
            "KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
        )

        # Add the patch sets of every surface section of the hemisphere to the input file
        for surf in surfs:
            patch_set = add_geomdl_nurbs_to_mesh(
                input_file,
                surf,
                material=mat,
                element_description=element_description,
            )

            input_file.add(patch_set)

        # Compare with the reference file
        compare_test_result(self, input_file.get_string(header=False))


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
