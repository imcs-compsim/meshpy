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
"""This script is used to test the mesh creation functions for NURBS."""

import numpy as np
import splinepy

from meshpy.core.mesh import Mesh
from meshpy.core.rotation import Rotation
from meshpy.four_c.material import MaterialSolid, MaterialStVenantKirchhoff
from meshpy.mesh_creation_functions.nurbs_generic import (
    add_geomdl_nurbs_to_mesh,
    add_splinepy_nurbs_to_mesh,
)
from meshpy.mesh_creation_functions.nurbs_geometries import (
    create_nurbs_brick,
    create_nurbs_cylindrical_shell_sector,
    create_nurbs_flat_plate_2d,
    create_nurbs_hemisphere_surface,
    create_nurbs_hollow_cylinder_segment_2d,
    create_nurbs_sphere_surface,
    create_nurbs_torus_surface,
)


def test_nurbs_hollow_cylinder_segment_2d(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of a two dimensional hollow cylinder segment."""

    # Create the surface of a quarter of a hollow cylinder
    surf_obj = create_nurbs_hollow_cylinder_segment_2d(
        1.74, 2.46, np.pi * 5 / 6, n_ele_u=2, n_ele_v=3
    )

    # Create mesh
    mesh = Mesh()

    # Add material
    mat = MaterialStVenantKirchhoff(youngs_modulus=50, nu=0.19, density=5.3e-7)

    # Create patch set
    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    patch_set = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj, material=mat, data=element_description
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_flat_plate_2d(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of a two dimensional flat plate."""

    # Create the surface of a flat plate
    surf_obj = create_nurbs_flat_plate_2d(0.75, 0.91, n_ele_u=2, n_ele_v=5)

    # Create mesh
    mesh = Mesh()

    # Add material
    mat = MaterialSolid(
        material_string="MAT_Kirchhoff_Love_shell",
        data={"YOUNG_MODULUS": 10.0, "POISSON_RATIO": 0.3, "THICKNESS": 0.05},
    )

    # Create patch set
    element_description = {"type": "SHELL_KIRCHHOFF_LOVE_NURBS", "GP": [3, 3]}
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj, material=mat, data=element_description
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_flat_plate_2d_splinepy(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of a two dimensional flat plate with splinepy."""

    # Create the surface of a flat plate
    n_ele_u = 2
    n_ele_v = 5
    surf_obj = splinepy.helpme.create.box(0.75, 0.91).nurbs
    surf_obj.elevate_degrees([0, 1])
    surf_obj.insert_knots(0, np.linspace(0, 1, n_ele_u + 1))
    surf_obj.insert_knots(1, np.linspace(0, 1, n_ele_v + 1))

    control_points_3d = np.zeros([len(surf_obj.control_points), 3])
    control_points_3d[:, :2] = surf_obj.control_points
    surf_obj.control_points = control_points_3d - 0.5 * np.array([0.75, 0.91, 0])

    # Create the shell mesh
    mesh = Mesh()
    mat = MaterialSolid(
        material_string="MAT_Kirchhoff_Love_shell",
        data={"YOUNG_MODULUS": 10.0, "POISSON_RATIO": 0.3, "THICKNESS": 0.05},
    )
    element_description = {"type": "SHELL_KIRCHHOFF_LOVE_NURBS", "GP": [3, 3]}
    patch_set = add_splinepy_nurbs_to_mesh(
        mesh, surf_obj, material=mat, data=element_description
    )
    mesh.add(patch_set)
    assert_results_equal(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_nurbs_flat_plate_2d"
        ),
        mesh,
    )


def test_nurbs_brick(assert_results_equal, get_corresponding_reference_file_path):
    """Test the creation of a brick."""

    # Create a brick
    vol_obj = create_nurbs_brick(1.5, 3.0, 2.4, n_ele_u=2, n_ele_v=3, n_ele_w=4)

    # Create mesh
    mesh = Mesh()

    # Add material
    mat = MaterialStVenantKirchhoff(youngs_modulus=710, nu=0.19, density=5.3e-7)

    # Create patch set
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        vol_obj,
        material=mat,
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_brick_splinepy(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of a brick with splinepy."""

    # Create a brick
    n_el_dim = [2, 3, 4]
    box_dimensions = [1.5, 3.0, 2.4]
    vol_obj = splinepy.helpme.create.box(*box_dimensions).nurbs
    vol_obj.elevate_degrees([0, 1, 2])
    for i_dim, n_el in enumerate(n_el_dim):
        vol_obj.insert_knots(i_dim, np.linspace(0, 1, n_el + 1))

    control_points_3d = np.zeros([len(vol_obj.control_points), 3])
    control_points_3d[:, :3] = vol_obj.control_points
    vol_obj.control_points = control_points_3d - 0.5 * np.array(box_dimensions)

    # Create mesh
    mesh = Mesh()

    # Add material
    mat = MaterialStVenantKirchhoff(youngs_modulus=710, nu=0.19, density=5.3e-7)

    # Create patch set
    patch_set = add_splinepy_nurbs_to_mesh(mesh, vol_obj, material=mat)

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_nurbs_brick"
        ),
        mesh,
    )


def test_nurbs_rotation_nurbs_surface(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the rotation of a NURBS mesh."""

    # Create the surface
    surf_obj = create_nurbs_hollow_cylinder_segment_2d(
        1.74, 2.46, np.pi * 3 / 4, n_ele_u=5, n_ele_v=2
    )

    # Create mesh
    mesh = Mesh()

    # Create patch set
    mat = MaterialStVenantKirchhoff(youngs_modulus=650, nu=0.20, density=4.2e-7)

    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    patch_set = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj, material=mat, data=element_description
    )

    mesh.add(patch_set)

    mesh.rotate(Rotation([1, 2, 3], np.pi * 7 / 6))

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_translate_nurbs_surface(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the translation of a NURBS surface mesh."""

    # Create the surface
    surf_obj = create_nurbs_flat_plate_2d(0.87, 1.35, n_ele_u=2, n_ele_v=3)

    # Create mesh
    mesh = Mesh()

    # Create patch set
    mat = MaterialStVenantKirchhoff(youngs_modulus=430, nu=0.10, density=4.2e-7)

    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    patch_set = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj, material=mat, data=element_description
    )

    mesh.add(patch_set)

    mesh.translate([-1.6, -2.3, 3.7])

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_cylindrical_shell_sector(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of a 3-dimensional cylindrical shell sector."""

    # Create the surface of a quarter of a hollow cylinder
    surf_obj = create_nurbs_cylindrical_shell_sector(
        2.3, np.pi / 3, 1.7, n_ele_u=3, n_ele_v=5
    )

    # Create mesh
    mesh = Mesh()

    # Add material
    mat = MaterialStVenantKirchhoff()

    # Create patch set
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        surf_obj,
        material=mat,
        data={"KINEM": "linear", "EAS": "none", "THICK": 1.0},
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_couple_nurbs_meshes(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the coupling of NURBS surface meshes."""

    # Create mesh
    mesh = Mesh()

    # Create the first surface object
    surf_obj_1 = create_nurbs_hollow_cylinder_segment_2d(
        0.65, 1.46, np.pi * 2 / 3, n_ele_u=3, n_ele_v=2
    )

    # Create first patch set
    mat = MaterialStVenantKirchhoff(youngs_modulus=430, nu=0.10, density=4.2e-7)

    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    patch_set_1 = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj_1, material=mat, data=element_description
    )

    mesh.add(patch_set_1)

    mesh.rotate(Rotation([0, 0, 1], np.pi / 3))

    # Create the second surface object
    surf_obj_2 = create_nurbs_hollow_cylinder_segment_2d(
        0.65, 1.46, np.pi / 3, n_ele_u=3, n_ele_v=2
    )

    patch_set_2 = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj_2, material=mat, data=element_description
    )

    mesh.add(patch_set_2)

    mesh.couple_nodes(reuse_matching_nodes=True)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_sphere_surface(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creating of the base patch of the surface of a sphere."""

    # Create mesh
    mesh = Mesh()

    # Create the base of a sphere
    surf_obj = create_nurbs_sphere_surface(1, n_ele_u=3, n_ele_v=2)

    # Create first patch set
    mat = MaterialStVenantKirchhoff()

    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    patch_set = add_geomdl_nurbs_to_mesh(
        mesh, surf_obj, material=mat, data=element_description
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_string_types(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creating of a NURBS with strings for the element and material
    definition."""

    # Create mesh
    mesh = Mesh()

    # Create the base of a sphere
    surf_obj = create_nurbs_flat_plate_2d(1, 3, n_ele_u=3, n_ele_v=2)

    # Create first patch set
    mat = MaterialStVenantKirchhoff()

    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        surf_obj,
        material=mat,
        data={"KINEM": "linear", "EAS": "none", "THICK": 1.0},
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_hemisphere_surface(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of the surface of a hemisphere."""

    # Create mesh
    mesh = Mesh()

    # Create the base of a sphere
    surfs = create_nurbs_hemisphere_surface(2.5, n_ele_uv=2)

    # Create first patch set
    mat = MaterialStVenantKirchhoff()

    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    # Add the patch sets of every surface section of the hemisphere to the input file
    for surf in surfs:
        patch_set = add_geomdl_nurbs_to_mesh(
            mesh, surf, material=mat, data=element_description
        )

        mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)


def test_nurbs_torus_surface(
    assert_results_equal, get_corresponding_reference_file_path
):
    """Test the creation of a torus."""

    # Create mesh
    mesh = Mesh()

    # Create the surface of a torus
    surfs = create_nurbs_torus_surface(1, 0.5, n_ele_u=2, n_ele_v=3)

    # Define material and element description
    mat = MaterialStVenantKirchhoff()

    element_description = {
        "KINEM": "linear",
        "EAS": "none",
        "THICK": 1.0,
        "STRESS_STRAIN": "plane_strain",
        "GP": [3, 3],
    }

    # Add the patch sets of every surface section of the torus to the input file
    for surf in surfs:
        patch_set = add_geomdl_nurbs_to_mesh(
            mesh, surf, material=mat, data=element_description
        )

        mesh.add(patch_set)

    # Compare with the reference file
    assert_results_equal(get_corresponding_reference_file_path(), mesh)
