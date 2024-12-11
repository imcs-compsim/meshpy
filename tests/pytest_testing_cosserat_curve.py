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
Test the functionality of the Cosserat curve module
"""

import os
import numpy as np
import pytest
import pyvista as pv
import json

from meshpy import mpy, InputFile, Rotation, Beam3rHerm2Line3, MaterialReissner
from meshpy.mesh_creation_functions import create_beam_mesh_helix

from meshpy.cosserat_curve.cosserat_curve import CosseratCurve
from meshpy.cosserat_curve.warping_along_cosserat_curve import (
    warp_mesh_along_curve,
    create_transform_boundary_conditions,
    get_mesh_transformation,
)

from .utils import (
    compare_string_tolerance,
    testing_input,
    testing_temp,
    get_pytest_test_name,
    compare_vtk_pytest,
    compare_test_result_pytest,
)


def load_cosserat_curve_from_file():
    """Load the centerline coordinates from the reference files and create the
    Cosserat curve."""
    coordinates = np.loadtxt(
        os.path.join(testing_input, "test_cosserat_curve_centerline.txt"),
        comments="#",
        delimiter=",",
        unpack=False,
    )
    return CosseratCurve(coordinates)


def create_beam_solid_input_file():
    """Create a beam and solid input file for testing purposes"""

    mpy.import_mesh_full = True
    mesh = InputFile(
        dat_file=os.path.join(testing_input, "test_cosserat_curve_mesh.dat")
    )
    create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        MaterialReissner(radius=0.05),
        [0, 0, 1],
        [0, 0, 0],
        [2, 0, 0],
        helix_angle=0.4,
        turns=3,
        n_el=5,
    )
    return mesh


def test_cosserat_curve_translate_and_rotate():
    """Test that a curve can be loaded, rotated and transformed"""

    curve = load_cosserat_curve_from_file()

    # Translate the curve so that the start is at the origin
    curve.translate(-curve.centerline_interpolation(5.0))

    # Rotate the curve around its center point
    pos_1, q_1 = curve.get_centerline_position_and_rotation(0.0)
    curve.rotate(q_1, origin=pos_1)

    # Get the points and rotations at certain points
    t = list(map(float, range(-10, 30, 5)))
    sol_half = curve.get_centerline_positions_and_rotations(
        t, factor=0.5, solve_ivp_kwargs={"atol": 1e-14, "rtol": 1e-12}
    )
    sol_full = curve.get_centerline_positions_and_rotations(t, factor=1.0)

    def load_compare(name):
        """Load the compare files and return a numpy array"""
        return np.loadtxt(
            os.path.join(testing_input, f"{get_pytest_test_name()}_{name}.txt")
        )

    assert np.allclose(sol_half[0], load_compare("pos_half_ref"), rtol=1e-8)
    assert np.allclose(sol_half[1], load_compare("q_half_ref"), rtol=1e-8)
    assert np.allclose(sol_full[0], load_compare("pos_full_ref"), rtol=1e-14)
    assert np.allclose(sol_full[1], load_compare("q_full_ref"), rtol=1e-14)


def test_cosserat_curve_vtk_representation():
    """Test the vtk representation of the Cosserat curve"""

    result_name = os.path.join(testing_temp, get_pytest_test_name() + ".vtu")
    curve = load_cosserat_curve_from_file()
    pv.UnstructuredGrid(curve.get_pyvista_polyline()).save(result_name)
    compare_vtk_pytest(
        os.path.join(testing_input, get_pytest_test_name() + ".vtu"),
        result_name,
    )


def test_cosserat_curve_project_point():
    """Test that the project point function works as expected"""

    # Load the curve
    curve = load_cosserat_curve_from_file()

    # Translate the curve so that the start is at the origin
    curve.translate(-curve.centerline_interpolation(0.0))

    # Check the projection results
    tol = 1e-14
    t_ref = 4.264045157204052
    assert t_ref == pytest.approx(curve.project_point([-5, 1, 1]), rel=tol)
    assert t_ref == pytest.approx(curve.project_point([-5, 1, 1], t0=2.0), rel=tol)
    assert t_ref == pytest.approx(curve.project_point([-5, 1, 1], t0=4.0), rel=tol)


def test_cosserat_mesh_transformation():
    """Test that the get_mesh_transformation function works as expected"""

    curve = load_cosserat_curve_from_file()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation.from_quaternion(rot)
    curve.translate(-pos)
    curve.translate([1, 2, 3])

    mesh = create_beam_solid_input_file()
    pos, rot = get_mesh_transformation(
        curve,
        mesh.nodes,
        origin=[0.5, 1.0, -1.0],
        reference_rotation=(
            Rotation([0, 0, 1], -0.5 * np.pi) * Rotation([0, 1, 0], -0.5 * np.pi)
        ),
        n_steps=3,
        solve_ivp_kwargs={"atol": 1e-14, "rtol": 1e-12},
    )

    # Save as json:
    # with open("name.json", "w") as f:
    #     json.dump(np_array.tolist(), f, indent=2)

    def load_result(name):
        with open(
            os.path.join(testing_input, f"{get_pytest_test_name()}_{name}.json"), "r"
        ) as f:
            return np.array(json.load(f))

    pos_ref = load_result("pos")
    rot_ref = load_result("rot")

    pos_np = np.array(pos)
    rot_np = np.array([[rotation.q for rotation in rot_step] for rot_step in rot])

    assert np.allclose(pos_ref, pos_np, rtol=1e-14)
    assert np.allclose(rot_ref, rot_np, rtol=1e-14)


def test_cosserat_curve_mesh_warp():
    """Warp a balloon along a centerline"""

    # Load the curve
    curve = load_cosserat_curve_from_file()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation.from_quaternion(rot)
    curve.translate(-pos)
    curve.translate([1, 2, 3])

    # Warp the mesh. The reference coordinate system is rotated such that z axis is the longitudinal direction,
    # and x and y are the first and second cross-section basis vectors respectively.
    mesh = create_beam_solid_input_file()
    warp_mesh_along_curve(
        mesh,
        curve,
        origin=[0.5, 1.0, -1.0],
        reference_rotation=(
            Rotation([0, 0, 1], -0.5 * np.pi) * Rotation([0, 1, 0], -0.5 * np.pi)
        ),
    )

    # Compare with the reference result
    compare_test_result_pytest(
        mesh.get_string(check_nox=False, header=False), rtol=1e-10
    )


def test_cosserat_curve_mesh_warp_transform_boundary_conditions():
    """Test the transform boundary creation function"""

    # Load the curve
    curve = load_cosserat_curve_from_file()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation.from_quaternion(rot)
    curve.translate(-pos)
    curve.translate([1, 2, 3])

    # Load the mesh
    mesh = create_beam_solid_input_file()

    # Apply the transform boundary conditions
    create_transform_boundary_conditions(
        mesh,
        curve,
        n_steps=3,
        origin=[2, 3, 0.5],
        reference_rotation=(
            Rotation([0, 0, 1], -0.5 * np.pi) * Rotation([0, 1, 0], -0.5 * np.pi)
        ),
        solve_ivp_kwargs={"atol": 1e-14, "rtol": 1e-12},
    )

    # Compare with the reference result
    compare_test_result_pytest(
        mesh.get_string(check_nox=False, header=False), rtol=1e-8, atol=1e-8
    )
