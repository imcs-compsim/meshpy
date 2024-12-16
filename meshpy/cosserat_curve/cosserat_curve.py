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
"""Define a Cosserat curve object that can be used to describe warping of
curve-like objects."""

from typing import Tuple

import numpy as np
import pyvista as pv
import quaternion
from numpy.typing import NDArray
from scipy import integrate, interpolate, optimize

from .. import mpy
from ..rotation import Rotation, rotate_coordinates, smallest_rotation


def get_piecewise_linear_arc_length_along_points(coordinates: np.ndarray) -> np.ndarray:
    """Return the accumulated distance between the points.

    Args
    ----
    coordinates:
        Array containing the point coordinates
    """

    n_points = len(coordinates)
    point_distance = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    point_arc_length = np.zeros(n_points)
    for i in range(1, n_points):
        point_arc_length[i] = point_arc_length[i - 1] + point_distance[i - 1]
    return point_arc_length


def get_spline_interpolation(
    coordinates: np.ndarray, point_arc_length: np.ndarray
) -> interpolate.BSpline:
    """Get a spline interpolation of the given points.

    Args
    ----
    coordinates:
        Array containing the point coordinates
    point_arc_length:
        Arc length for each coordinate

    Return
    ----
    centerline_interpolation:
        The spline interpolation object
    """

    # Interpolate coordinates along arc length
    centerline_interpolation = interpolate.make_interp_spline(
        point_arc_length, coordinates
    )
    return centerline_interpolation


def get_quaternions_along_curve(
    centerline: interpolate.BSpline, point_arc_length: np.ndarray
) -> NDArray[quaternion.quaternion]:
    """Get the quaternions along the curve based on smallest rotation mappings.

    The initial rotation will be calculated based on the largest projection of the initial tangent
    onto the cartesian basis vectors.

    Args
    ----
    centerline:
        A function that returns the centerline position for a parameter coordinate t
    point_arc_length:
        Array of parameter coordinates for which the quaternions should be calculated
    """

    centerline_interpolation_derivative = centerline.derivative()

    def basis(i):
        """Return the i-th Cartesian basis vector."""
        basis = np.zeros([3])
        basis[i] = 1.0
        return basis

    # Get the reference rotation
    t0 = centerline_interpolation_derivative(point_arc_length[0])
    min_projection = np.argmin(np.abs([np.dot(basis(i), t0) for i in range(3)]))
    last_rotation = Rotation.from_basis(t0, basis(min_projection))

    # Get the rotation vectors along the curve. They are calculated with smallest rotation mappings.
    n_points = len(point_arc_length)
    quaternions = np.zeros(n_points, dtype=quaternion.quaternion)
    quaternions[0] = last_rotation.q
    for i in range(1, n_points):
        rotation = smallest_rotation(
            last_rotation,
            centerline_interpolation_derivative(point_arc_length[i]),
        )
        quaternions[i] = rotation.q
        last_rotation = rotation
    return quaternions


def get_relative_distance_and_rotations(
    coordinates: np.ndarray, quaternions: NDArray[quaternion.quaternion]
) -> Tuple[np.ndarray, NDArray[quaternion.quaternion], NDArray[quaternion.quaternion]]:
    """Get relative distances and rotations that can be used to evaluate
    "intermediate" states of the Cosserat curve."""

    n_points = len(coordinates)
    relative_distances = np.zeros(n_points - 1)
    relative_distances_rotation = np.zeros(n_points - 1, dtype=quaternion.quaternion)
    relative_rotations = np.zeros(n_points - 1, dtype=quaternion.quaternion)

    for i_segment in range(n_points - 1):
        relative_distance = coordinates[i_segment + 1] - coordinates[i_segment]
        relative_distance_local = quaternion.rotate_vectors(
            quaternions[i_segment].conjugate(), relative_distance
        )
        relative_distances[i_segment] = np.linalg.norm(relative_distance_local)

        smallest_relative_rotation_onto_distance = smallest_rotation(
            Rotation(),
            relative_distance_local,
        )
        relative_distances_rotation[i_segment] = quaternion.from_float_array(
            smallest_relative_rotation_onto_distance.q
        )
        relative_rotations[i_segment] = (
            quaternions[i_segment + 1] * quaternions[i_segment].conjugate()
        )

    return relative_distances, relative_distances_rotation, relative_rotations


class CosseratCurve(object):
    """Represent a Cosserat curve in space."""

    def __init__(self, point_coordinates: np.ndarray):
        """Initialize the Cosserat curve based on points in 3D space.

        Args
        ----
        point_coordinates:
            Array containing the point coordinates
        """

        self.coordinates = point_coordinates.copy()
        self.n_points = len(self.coordinates)

        # Interpolate coordinates along piece wise linear arc length
        point_arc_length_piecewise_linear = (
            get_piecewise_linear_arc_length_along_points(self.coordinates)
        )
        centerline_interpolation_piecewise_linear = get_spline_interpolation(
            self.coordinates, point_arc_length_piecewise_linear
        )
        centerline_interpolation_piecewise_linear_p = (
            centerline_interpolation_piecewise_linear.derivative(1)
        )

        def ds(t):
            """Arc length along interpolated spline."""
            return np.linalg.norm(centerline_interpolation_piecewise_linear_p(t))

        # Integrate the arc length along the interpolated centerline, this will result
        # in a more accurate centerline arc length
        self.point_arc_length = [0.0]
        for i in range(len(point_arc_length_piecewise_linear) - 1):
            self.point_arc_length.append(
                self.point_arc_length[-1]
                + integrate.quad(
                    ds,
                    point_arc_length_piecewise_linear[i],
                    point_arc_length_piecewise_linear[i + 1],
                )[0]
            )

        # Set the interpolation of the (positional) centerline
        self.set_centerline_interpolation()

        # Get the quaternions along the centerline based on smallest rotation mappings
        self.quaternions = get_quaternions_along_curve(
            self.centerline_interpolation, self.point_arc_length
        )

        # Get the relative quantities used to warp the curve
        (
            self.relative_distances,
            self.relative_distances_rotation,
            self.relative_rotations,
        ) = get_relative_distance_and_rotations(self.coordinates, self.quaternions)

    def get_curvature_function(self, *, factor: float = 1.0):
        """Get a function that returns the curvature along the centerline.

        Args
        ----
        factor: double
            Scaling factor for the curvature
        """

        centerline_interpolation_p = self.centerline_interpolation.derivative(1)
        centerline_interpolation_pp = self.centerline_interpolation.derivative(2)

        def curvature(t):
            """Get the curvature along the curve."""
            rp = centerline_interpolation_p
            rpp = centerline_interpolation_pp
            return factor * np.cross(rp(t), rpp(t)) / np.dot(rp(t), rp(t))

        return curvature

    def set_centerline_interpolation(self):
        """Set the interpolation of the centerline based on the coordinates and
        arc length stored in this object."""
        self.centerline_interpolation = get_spline_interpolation(
            self.coordinates, self.point_arc_length
        )

    def translate(self, vector):
        """Translate the curve by the given vector."""

        self.coordinates += vector
        self.set_centerline_interpolation()

    def rotate(self, rotation: Rotation, *, origin=None):
        """Rotate the curve and the quaternions."""

        self.quaternions = quaternion.from_float_array(rotation.q) * self.quaternions
        self.coordinates = rotate_coordinates(self.coordinates, rotation, origin=origin)
        self.set_centerline_interpolation()

    def get_centerline_position_and_rotation(
        self, arc_length: float, **kwargs
    ) -> Tuple[np.ndarray, NDArray[quaternion.quaternion]]:
        """Return the position and rotation at a given centerline arc
        length."""
        pos, rot = self.get_centerline_positions_and_rotations([arc_length])
        return pos[0], rot[0]

    def get_centerline_positions_and_rotations(
        self, points_on_arc_length, *, factor=1.0
    ) -> Tuple[np.ndarray, NDArray[quaternion.quaternion]]:
        """Return the position and rotation at given centerline arc lengths.

        If the points are outside of the valid interval, a linear extrapolation will be
        performed for the displacements and the rotations will be held constant.

        Args
        ----
        points_on_arc_length: list(float)
            A sorted list with the arc lengths along the curve centerline
        factor: float
            Factor to scale the curvature along the curve.
                factor == 1
                    Use the default positions and the triads obtained via a smallest rotation mapping
                factor < 1
                    Integrate (piecewise constant as evaluated with get_relative_distance_and_rotations)
                    the scaled curvature of the curve to obtain a intuitive wrapping
        """

        # Get the points that are within the arc length of the given curve.
        points_on_arc_length = np.array(points_on_arc_length)
        points_in_bounds = np.logical_and(
            points_on_arc_length > self.point_arc_length[0],
            points_on_arc_length < self.point_arc_length[-1],
        )
        index_in_bound = np.where(points_in_bounds == True)[0]
        index_out_of_bound = np.where(points_in_bounds == False)[0]
        points_on_arc_length_in_bound = [
            self.point_arc_length[0],
            *points_on_arc_length[index_in_bound],
            self.point_arc_length[-1],
        ]

        if factor < (1.0 - mpy.eps_quaternion):
            coordinates = np.zeros_like(self.coordinates)
            quaternions = np.zeros_like(self.quaternions)
            coordinates[0] = self.coordinates[0]
            quaternions[0] = self.quaternions[0]
            for i_segment in range(self.n_points - 1):
                relative_distance_rotation = quaternion.slerp_evaluate(
                    quaternion.quaternion(1),
                    self.relative_distances_rotation[i_segment],
                    factor,
                )
                # In the initial configuration (factor=0) we get a straight curve, so we need
                # to use the arc length here. In the final configuration (factor=1) we want to
                # exactly recover the input points, so we need the piecewise linear distance.
                # Between them, we interpolate.
                relative_distance = (factor * self.relative_distances[i_segment]) + (
                    1.0 - factor
                ) * (
                    self.point_arc_length[i_segment + 1]
                    - self.point_arc_length[i_segment]
                )
                coordinates[i_segment + 1] = (
                    quaternion.rotate_vectors(
                        quaternions[i_segment] * relative_distance_rotation,
                        [relative_distance, 0, 0],
                    )
                    + coordinates[i_segment]
                )
                quaternions[i_segment + 1] = (
                    quaternion.slerp_evaluate(
                        quaternion.quaternion(1),
                        self.relative_rotations[i_segment],
                        factor,
                    )
                    * quaternions[i_segment]
                )
        else:
            coordinates = self.coordinates
            quaternions = self.quaternions

        sol_r = np.zeros([len(points_on_arc_length_in_bound), 3])
        sol_q = np.zeros(
            len(points_on_arc_length_in_bound), dtype=quaternion.quaternion
        )
        for i_point, centerline_arc_length in enumerate(points_on_arc_length_in_bound):
            if (
                centerline_arc_length >= self.point_arc_length[0]
                and centerline_arc_length <= self.point_arc_length[-1]
            ):
                for i in range(1, self.n_points):
                    centerline_index = i - 1
                    if self.point_arc_length[i] > centerline_arc_length:
                        break

                # Get the two rotation vectors and arc length values
                arc_lengths = self.point_arc_length[
                    centerline_index : centerline_index + 2
                ]
                q1 = quaternions[centerline_index]
                q2 = quaternions[centerline_index + 1]

                # Linear interpolate the arc length
                xi = (centerline_arc_length - arc_lengths[0]) / (
                    arc_lengths[1] - arc_lengths[0]
                )

                # Perform a spline interpolation for the positions and a slerp
                # interpolation for the rotations
                sol_r[i_point] = get_spline_interpolation(
                    coordinates, self.point_arc_length
                )(centerline_arc_length)
                sol_q[i_point] = quaternion.as_float_array(
                    quaternion.slerp_evaluate(q1, q2, xi)
                )
            else:
                raise ValueError("Centerline value out of bounds")

        # Set the already computed results in the final data structures
        sol_r_final = np.zeros([len(points_on_arc_length), 3])
        sol_q_final = np.zeros(len(points_on_arc_length), dtype=quaternion.quaternion)
        if len(index_in_bound) > 0:
            sol_r_final[index_in_bound] = sol_r[index_in_bound - index_in_bound[0] + 1]
            sol_q_final[index_in_bound] = sol_q[index_in_bound - index_in_bound[0] + 1]

        # Perform the extrapolation at both ends of the curve
        for i in index_out_of_bound:
            arc_length = points_on_arc_length[i]
            if arc_length <= self.point_arc_length[0]:
                index = 0
            elif arc_length >= self.point_arc_length[-1]:
                index = -1
            else:
                raise ValueError("Should not happen")

            length = arc_length - self.point_arc_length[index]
            r = sol_r[index]
            q = sol_q[index]
            sol_r_final[i] = r + Rotation.from_quaternion(
                quaternion.as_float_array(q)
            ) * [length, 0, 0]
            sol_q_final[i] = q

        return sol_r_final, sol_q_final

    def project_point(self, p, t0=None) -> float:
        """Project a point to the curve, return the parameter coordinate for
        the projection point."""

        centerline_interpolation_p = self.centerline_interpolation.derivative(1)
        centerline_interpolation_pp = self.centerline_interpolation.derivative(2)

        def f(t):
            """Function to find the root of."""
            r = self.centerline_interpolation(t)
            rp = centerline_interpolation_p(t)
            return np.dot(r - p, rp)

        def fp(t):
            """Derivative of the Function to find the root of."""
            r = self.centerline_interpolation(t)
            rp = centerline_interpolation_p(t)
            rpp = centerline_interpolation_pp(t)
            return np.dot(rp, rp) + np.dot(r - p, rpp)

        if t0 is None:
            t0 = 0.0

        return optimize.newton(f, t0, fprime=fp)

    def get_pyvista_polyline(self) -> pv.PolyData:
        """Create a pyvista (vtk) representation of the curve with the
        evaluated triad basis vectors."""

        poly_line = pv.PolyData()
        poly_line.points = self.coordinates
        cell = np.arange(0, self.n_points, dtype=int)
        cell = np.insert(cell, 0, self.n_points)
        poly_line.lines = cell

        rotation_matrices = np.zeros((len(self.quaternions), 3, 3))
        for i_quaternion, q in enumerate(self.quaternions):
            R = quaternion.as_rotation_matrix(q)
            rotation_matrices[i_quaternion] = R

        for i_dir in range(3):
            poly_line.point_data.set_array(
                rotation_matrices[:, :, i_dir], f"base_vector_{i_dir + 1}"
            )

        return poly_line

    def write_vtk(self, path) -> None:
        """Save a vtk representation of the curve."""
        self.get_pyvista_polyline().save(path)
