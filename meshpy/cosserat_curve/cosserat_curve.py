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
Define a Cosserat curve object that can be used to describe warping of curve-like
objects
"""

import numpy as np
from scipy import integrate, optimize, interpolate
import quaternion
import pyvista as pv

from .. import mpy
from ..rotation import Rotation, add_rotations, rotate_coordinates, smallest_rotation


def get_piecewise_linear_arc_length_along_points(coordinates: np.array):
    """Return the accumulated distance between the points

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


def get_spline_interpolation(coordinates: np.array, point_arc_length: np.array):
    """Get a spline interpolation of the given points

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


def get_quaternions_along_curve(centerline, point_arc_length):
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
        """Return the i-th Cartesian basis vector"""
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


class CosseratCurve(object):
    """Represent a Cosserat curve in space"""

    def __init__(self, point_coordinates):
        """Initialize the Cosserat curve based on points in 3D space

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

    def get_curvature_function(self, *, factor=1.0):
        """Get a function that returns the curvature along the centerline

        Args
        ----
        factor: double
            Scaling factor for the curvature
        """

        centerline_interpolation_p = self.centerline_interpolation.derivative(1)
        centerline_interpolation_pp = self.centerline_interpolation.derivative(2)

        def curvature(t):
            """Get the curvature along the curve"""
            rp = centerline_interpolation_p
            rpp = centerline_interpolation_pp
            return factor * np.cross(rp(t), rpp(t)) / np.dot(rp(t), rp(t))

        return curvature

    def set_centerline_interpolation(self):
        """Set the interpolation of the centerline based on the coordinates and arc length stored in this object"""
        self.centerline_interpolation = get_spline_interpolation(
            self.coordinates, self.point_arc_length
        )

    def translate(self, vector):
        """Translate the curve by the given vector"""

        self.coordinates += vector
        self.set_centerline_interpolation()

    def rotate(self, rotation, *, origin=None):
        """Rotate the curve and the quaternions"""

        self.quaternions = quaternion.from_float_array(rotation.q) * self.quaternions
        self.coordinates = rotate_coordinates(self.coordinates, rotation, origin=origin)
        self.set_centerline_interpolation()

    def get_centerline_position_and_rotation(self, arc_length, **kwargs):
        """Return the position and rotation at a given centerline arc length"""
        pos, rot = self.get_centerline_positions_and_rotations([arc_length])
        return pos[0], rot[0]

    def get_centerline_positions_and_rotations(
        self,
        points_on_arc_length,
        *,
        factor=1.0,
        solve_ivp_kwargs={"atol": 1e-08, "rtol": 1e-08},
    ):
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
                    Integrate the (scaled by the factor) curvature of the curve to obtain a intuitive unwrapping
                The results will have a small discontinuity between a factor close to 1 and exactly one. This
                is due to the different evaluation of the centerline and the rotations. For an exact integration
                of the ivp, the centerline positions will match, however there might still be a difference in
                the rotations. For practical purposes this does not affect the usability of the results.
        solve_ivp_kwargs: dict
            Keyword arguments passed scipy.integrate.solve_ivp
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
            centerline_interpolation_p = self.centerline_interpolation.derivative(1)

            # Integrate the curvature along the centerline. The curvature is multiplied with the factor
            # to allow for a consistent "warping" of the curve.
            curvature = self.get_curvature_function(factor=factor)

            def rhs(t, y):
                """Right hand side of the differential equation, see meshpy:utility/rotation.wls"""
                q = y[:4]
                rotation_angular_vel = quaternion.from_float_array(
                    [0.0, *(curvature(t))]
                )
                rotation_increment = (
                    rotation_angular_vel * quaternion.from_float_array(q)
                ) * 0.5
                dr = np.linalg.norm(
                    centerline_interpolation_p(t)
                ) * quaternion.rotate_vectors(quaternion.from_float_array(q), [1, 0, 0])
                return [*quaternion.as_float_array(rotation_increment), *dr]

            # Integrate the position and rotation along the centerline
            # The accuracy can be improved with the tolerance parameters
            sol_ivp = integrate.solve_ivp(
                rhs,
                [self.point_arc_length[0], self.point_arc_length[-1]],
                [
                    *(quaternion.as_float_array(self.quaternions[0])),
                    *(self.centerline_interpolation(0)),
                ],
                t_eval=points_on_arc_length_in_bound,
                **solve_ivp_kwargs,
            )
            sol_y = np.transpose((sol_ivp.y))
            sol_r = sol_y[:, 4:]
            sol_q = quaternion.as_quat_array(sol_y[:, :4])
        else:
            # Do a slerp interpolation of quaternions for the given points
            sol_r = np.zeros([len(points_on_arc_length_in_bound), 3])
            sol_q = np.zeros(
                len(points_on_arc_length_in_bound), dtype=quaternion.quaternion
            )
            for i_point, centerline_arc_length in enumerate(
                points_on_arc_length_in_bound
            ):
                if (
                    centerline_arc_length >= self.point_arc_length[0]
                    and centerline_arc_length <= self.point_arc_length[-1]
                ):
                    for i in range(1, self.n_points):
                        centerline_index = i - 1
                        if self.point_arc_length[i] > centerline_arc_length:
                            break

                    # Get the two rotation vectors and arc length values
                    arc_length = self.point_arc_length[
                        centerline_index : centerline_index + 2
                    ]
                    q1 = self.quaternions[centerline_index]
                    q2 = self.quaternions[centerline_index + 1]

                    # Linear interpolate the arc length
                    xi = (centerline_arc_length - arc_length[0]) / (
                        arc_length[1] - arc_length[0]
                    )

                    # Perform a slerp interpolation of the rotation
                    rotation = Rotation.from_quaternion(
                        quaternion.as_float_array(quaternion.slerp_evaluate(q1, q2, xi))
                    )

                    sol_r[i_point] = self.centerline_interpolation(
                        centerline_arc_length
                    )
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

    def project_point(self, p, t0=None):
        """Project a point to the curve"""

        centerline_interpolation_p = self.centerline_interpolation.derivative(1)
        centerline_interpolation_pp = self.centerline_interpolation.derivative(2)

        def f(t):
            """Function to find the root of"""
            r = self.centerline_interpolation(t)
            rp = centerline_interpolation_p(t)
            return np.dot(r - p, rp)

        def fp(t):
            """Derivative of the Function to find the root of"""
            r = self.centerline_interpolation(t)
            rp = centerline_interpolation_p(t)
            rpp = centerline_interpolation_pp(t)
            return np.dot(rp, rp) + np.dot(r - p, rpp)

        if t0 is None:
            t0 = 0.0

        return optimize.newton(f, t0, fprime=fp)

    def get_pyvista_polyline(self) -> pv.PolyData:
        """Create a pyvista (vtk) representation of the curve with the
        evaluated triad basis vectors"""

        poly_line = pv.PolyData()
        poly_line.points = self.coordinates
        cell = np.arange(0, self.n_points, dtype=int)
        cell = np.insert(cell, 0, self.n_points)
        poly_line.lines = cell

        base = [[], [], []]
        for q in self.quaternions:
            R = Rotation.from_quaternion(
                quaternion.as_float_array(q)
            ).get_rotation_matrix()
            for i_dir in range(3):
                base[i_dir].append(R[:, i_dir])

        for i_dir in range(3):
            poly_line.point_data.set_array(base[i_dir], f"base_vector_{i_dir + 1}")

        return poly_line

    def write_vtk(self, path):
        """Save a vtk representation of the curve"""
        self.get_pyvista_polyline().save(path)
