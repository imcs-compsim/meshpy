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

from meshpy.cosserat_curve.cosserat_curve import CosseratCurve
from meshpy.cosserat_curve.warp_mesh_along_curve import warp_mesh_along_curve
from meshpy import mpy, InputFile, Rotation, Beam3rHerm2Line3, MaterialReissner
from meshpy.mesh_creation_functions import create_beam_mesh_line
from .utils import compare_string_tolerance, testing_input


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


def test_translate_and_rotate():
    """Test that a curve can be loaded, rotated and transformed"""

    curve = load_cosserat_curve_from_file()

    # Translate the curve so that the start is at the origin
    curve.translate(-curve.centerline_interpolation(5.0))

    # Rotate the curve around its center point
    pos_1, q_1 = curve.get_centerline_position_and_rotation(0.0)
    curve.rotate(q_1, origin=pos_1)

    # Get the points and rotations at certain points
    t = list(map(float, range(-10, 30, 5)))
    sol_half = curve.get_centerline_positions_and_rotations(t, factor=0.5)
    sol_full = curve.get_centerline_positions_and_rotations(t, factor=1.0)

    def load_compare(name):
        """Load the compare files and return a numpy array"""
        return np.loadtxt(
            os.path.join(
                testing_input,
                f"test_cosserat_curve_{name}.txt",
            )
        )

    assert np.allclose(
        sol_half[0], load_compare("translate_and_rotate_pos_half_ref"), rtol=1e-14
    )
    assert np.allclose(
        sol_half[1], load_compare("translate_and_rotate_q_half_ref"), rtol=1e-14
    )
    assert np.allclose(
        sol_full[0], load_compare("translate_and_rotate_pos_full_ref"), rtol=1e-14
    )
    assert np.allclose(
        sol_full[1], load_compare("translate_and_rotate_q_full_ref"), rtol=1e-14
    )


def test_project_point():
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


def test_warp_mesh():
    """Warp a balloon along a centerline"""

    # Load the curve
    curve = load_cosserat_curve_from_file()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation(rot)
    curve.translate(-pos)
    curve.rotate(Rotation([0, 1, 0], -0.5 * np.pi) * rot.inv())

    # Load the mesh
    mpy.import_mesh_full = True
    mesh = InputFile(
        dat_file=os.path.join(testing_input, "test_cosserat_curve_balloon_mesh.dat")
    )

    # Add a beam mesh
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, MaterialReissner(), [1, 2, 0], [1, 2, 17]
    )

    # Warp the mesh. The reference coordinate system is rotated such that z axis is the longitudinal direction,
    # and x and y are the first and second cross-section basis vectors respectively.
    warp_mesh_along_curve(
        mesh,
        curve,
        reference_rotation=Rotation([0, 0, 1], -0.5 * np.pi)
        * Rotation([0, 1, 0], -0.5 * np.pi),
    )

    # Compare the results
    input_file_string = "\n".join(mesh.get_dat_lines(header=False, dat_header=False))
    with open(
        os.path.join(testing_input, "test_cosserat_curve_balloon_mesh_warp.dat")
    ) as f:
        data = f.read()
    # Todo: Once we switched to pytest, lets put the general compare results function here
    assert compare_string_tolerance(data, input_file_string, rtol=1e-10)
