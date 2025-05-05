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
"""Define a Cosserat curve object that can be used to describe warping of
curve-like objects."""

from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
import pyvista as _pv
import quaternion as _quaternion
from numpy.typing import NDArray as _NDArray
from scipy import integrate as _integrate
from scipy import interpolate as _interpolate
from scipy import optimize as _optimize

from meshpy.core.conf import mpy as _mpy
from meshpy.core.rotation import Rotation as _Rotation
from meshpy.core.rotation import rotate_coordinates as _rotate_coordinates
from meshpy.core.rotation import smallest_rotation as _smallest_rotation


def get_piecewise_linear_arc_length_along_points(
    coordinates: _np.ndarray,
) -> _np.ndarray:
    """Return the accumulated distance between the points.

    Args
    ----
    coordinates:
        Array containing the point coordinates
    """

    n_points = len(coordinates)
    point_distance = _np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    point_arc_length = _np.zeros(n_points)
    for i in range(1, n_points):
        point_arc_length[i] = point_arc_length[i - 1] + point_distance[i - 1]
    return point_arc_length


def get_spline_interpolation(
    coordinates: _np.ndarray, point_arc_length: _np.ndarray
) -> _interpolate.BSpline:
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
    centerline_interpolation = _interpolate.make_interp_spline(
        point_arc_length, coordinates
    )
    return centerline_interpolation


def get_quaternions_along_curve(
    centerline: _interpolate.BSpline, point_arc_length: _np.ndarray
) -> _NDArray[_quaternion.quaternion]:
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
        basis = _np.zeros([3])
        basis[i] = 1.0
        return basis

    # Get the reference rotation
    t0 = centerline_interpolation_derivative(point_arc_length[0])
    min_projection = _np.argmin(_np.abs([_np.dot(basis(i), t0) for i in range(3)]))
    last_rotation = _Rotation.from_basis(t0, basis(min_projection))

    # Get the rotation vectors along the curve. They are calculated with smallest rotation mappings.
    n_points = len(point_arc_length)
    quaternions = _np.zeros(n_points, dtype=_quaternion.quaternion)
    quaternions[0] = last_rotation.q
    for i in range(1, n_points):
        rotation = _smallest_rotation(
            last_rotation,
            centerline_interpolation_derivative(point_arc_length[i]),
        )
        quaternions[i] = rotation.q
        last_rotation = rotation
    return quaternions


def get_relative_distance_and_rotations(
    coordinates: _np.ndarray, quaternions: _NDArray[_quaternion.quaternion]
) -> _Tuple[
    _np.ndarray, _NDArray[_quaternion.quaternion], _NDArray[_quaternion.quaternion]
]:
    """Get relative distances and rotations that can be used to evaluate
    "intermediate" states of the Cosserat curve."""

    n_points = len(coordinates)
    relative_distances = _np.zeros(n_points - 1)
    relative_distances_rotation = _np.zeros(n_points - 1, dtype=_quaternion.quaternion)
    relative_rotations = _np.zeros(n_points - 1, dtype=_quaternion.quaternion)

    for i_segment in range(n_points - 1):
        relative_distance = coordinates[i_segment + 1] - coordinates[i_segment]
        relative_distance_local = _quaternion.rotate_vectors(
            quaternions[i_segment].conjugate(), relative_distance
        )
        relative_distances[i_segment] = _np.linalg.norm(relative_distance_local)

        smallest_relative_rotation_onto_distance = _smallest_rotation(
            _Rotation(),
            relative_distance_local,
        )
        relative_distances_rotation[i_segment] = _quaternion.from_float_array(
            smallest_relative_rotation_onto_distance.q
        )

        relative_rotations[i_segment] = (
            quaternions[i_segment].conjugate() * quaternions[i_segment + 1]
        )

    return relative_distances, relative_distances_rotation, relative_rotations


class CosseratCurve(object):
    """Represent a Cosserat curve in space."""

    def __init__(
        self,
        point_coordinates: _np.ndarray,
        *,
        starting_triad_guess: _Optional[_Rotation] = None,
    ):
        """Initialize the Cosserat curve based on points in 3D space.

        Args:
            point_coordinates: Array containing the point coordinates
            starting_triad_guess: Optional initial guess for the starting triad.
                If provided, this introduces a constant twist angle along the curve.
                The twist angle is computed between:
                - The given starting guess triad, and
                - The automatically calculated triad, rotated onto the first basis vector
                  of the starting guess triad using the smallest rotation.
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
            return _np.linalg.norm(centerline_interpolation_piecewise_linear_p(t))

        # Integrate the arc length along the interpolated centerline, this will result
        # in a more accurate centerline arc length
        self.point_arc_length = _np.zeros(self.n_points)
        for i in range(len(point_arc_length_piecewise_linear) - 1):
            self.point_arc_length[i + 1] = (
                self.point_arc_length[i]
                + _integrate.quad(
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

        # Check if we have to apply a twist for the rotations
        if starting_triad_guess is not None:
            first_rotation = _Rotation.from_quaternion(
                _quaternion.as_float_array(self.quaternions[0])
            )
            starting_triad_e1 = starting_triad_guess * [1, 0, 0]
            if _np.dot(first_rotation * [1, 0, 0], starting_triad_e1) < 0.5:
                raise ValueError(
                    "The angle between the first basis vectors of the guess triad you"
                    " provided and the automatically calculated one is too large,"
                    " please check your input data."
                )
            smallest_rotation_to_guess_tangent = _smallest_rotation(
                first_rotation, starting_triad_e1
            )
            relative_rotation = (
                smallest_rotation_to_guess_tangent.inv() * starting_triad_guess
            )
            psi = relative_rotation.get_rotation_vector()
            if _np.linalg.norm(psi[1:]) > _mpy.eps_quaternion:
                raise ValueError(
                    "The twist angle can not be extracted as the relative rotation is not plane!"
                )
            twist_angle = psi[0]
            self.twist(twist_angle)

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

    def rotate(self, rotation: _Rotation, *, origin=None):
        """Rotate the curve and the quaternions."""

        self.quaternions = _quaternion.from_float_array(rotation.q) * self.quaternions
        self.coordinates = _rotate_coordinates(
            self.coordinates, rotation, origin=origin
        )
        self.set_centerline_interpolation()

    def twist(self, twist_angle: float) -> None:
        """Apply a constant twist rotation along the Cosserat curve.

        Args:
            twist_angle: The rotation angle (in radiants).
        """

        material_twist_rotation = _Rotation([1, 0, 0], twist_angle)
        # TODO: use numpy array functions for the following operations.
        for i in range(len(self.quaternions)):
            self.quaternions[i] = self.quaternions[i] * _quaternion.from_float_array(
                material_twist_rotation.q
            )
        for i, (relative_distances_rotation, relative_rotation) in enumerate(
            zip(self.relative_distances_rotation, self.relative_rotations)
        ):
            relative_distances_rotation = _Rotation.from_quaternion(
                _quaternion.as_float_array(relative_distances_rotation)
            )
            relative_rotation = _Rotation.from_quaternion(
                _quaternion.as_float_array(relative_rotation)
            )

            new_relative_distances_rotation = (
                material_twist_rotation.inv()
                * relative_distances_rotation
                * material_twist_rotation
            )
            new_relative_rotation = (
                material_twist_rotation.inv()
                * relative_rotation
                * material_twist_rotation
            )
            self.relative_distances_rotation[i] = _quaternion.from_float_array(
                new_relative_distances_rotation.q
            )
            self.relative_rotations[i] = _quaternion.from_float_array(
                new_relative_rotation.q
            )

    def get_centerline_position_and_rotation(
        self, arc_length: float, **kwargs
    ) -> _Tuple[_np.ndarray, _NDArray[_quaternion.quaternion]]:
        """Return the position and rotation at a given centerline arc
        length."""
        pos, rot = self.get_centerline_positions_and_rotations([arc_length], **kwargs)
        return pos[0], rot[0]

    def get_centerline_positions_and_rotations(
        self, points_on_arc_length, *, factor=1.0
    ) -> _Tuple[_np.ndarray, _NDArray[_quaternion.quaternion]]:
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
        points_on_arc_length = _np.asarray(points_on_arc_length)
        points_in_bounds = _np.logical_and(
            points_on_arc_length > self.point_arc_length[0],
            points_on_arc_length < self.point_arc_length[-1],
        )
        index_in_bound = _np.where(points_in_bounds == True)[0]
        index_out_of_bound = _np.where(points_in_bounds == False)[0]
        points_on_arc_length_in_bound = [
            self.point_arc_length[0],
            *points_on_arc_length[index_in_bound],
            self.point_arc_length[-1],
        ]

        if factor < (1.0 - _mpy.eps_quaternion):
            coordinates = _np.zeros_like(self.coordinates)
            quaternions = _np.zeros_like(self.quaternions)
            coordinates[0] = self.coordinates[0]
            quaternions[0] = self.quaternions[0]
            for i_segment in range(self.n_points - 1):
                relative_distance_rotation = _quaternion.slerp_evaluate(
                    _quaternion.quaternion(1),
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
                    _quaternion.rotate_vectors(
                        quaternions[i_segment] * relative_distance_rotation,
                        [relative_distance, 0, 0],
                    )
                    + coordinates[i_segment]
                )
                quaternions[i_segment + 1] = quaternions[
                    i_segment
                ] * _quaternion.slerp_evaluate(
                    _quaternion.quaternion(1),
                    self.relative_rotations[i_segment],
                    factor,
                )
        else:
            coordinates = self.coordinates
            quaternions = self.quaternions

        sol_r = _np.zeros([len(points_on_arc_length_in_bound), 3])
        sol_q = _np.zeros(
            len(points_on_arc_length_in_bound), dtype=_quaternion.quaternion
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
                sol_q[i_point] = _quaternion.as_float_array(
                    _quaternion.slerp_evaluate(q1, q2, xi)
                )
            else:
                raise ValueError("Centerline value out of bounds")

        # Set the already computed results in the final data structures
        sol_r_final = _np.zeros([len(points_on_arc_length), 3])
        sol_q_final = _np.zeros(len(points_on_arc_length), dtype=_quaternion.quaternion)
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
            sol_r_final[i] = r + _Rotation.from_quaternion(
                _quaternion.as_float_array(q)
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
            return _np.dot(r - p, rp)

        def fp(t):
            """Derivative of the Function to find the root of."""
            r = self.centerline_interpolation(t)
            rp = centerline_interpolation_p(t)
            rpp = centerline_interpolation_pp(t)
            return _np.dot(rp, rp) + _np.dot(r - p, rpp)

        if t0 is None:
            t0 = 0.0

        return _optimize.newton(f, t0, fprime=fp)

    def get_pyvista_polyline(self) -> _pv.PolyData:
        """Create a pyvista (vtk) representation of the curve with the
        evaluated triad basis vectors."""

        poly_line = _pv.PolyData()
        poly_line.points = self.coordinates
        cell = _np.arange(0, self.n_points, dtype=int)
        cell = _np.insert(cell, 0, self.n_points)
        poly_line.lines = cell

        rotation_matrices = _np.zeros((len(self.quaternions), 3, 3))
        for i_quaternion, q in enumerate(self.quaternions):
            R = _quaternion.as_rotation_matrix(q)
            rotation_matrices[i_quaternion] = R

        for i_dir in range(3):
            poly_line.point_data.set_array(
                rotation_matrices[:, :, i_dir], f"base_vector_{i_dir + 1}"
            )

        return poly_line

    def write_vtk(self, path) -> None:
        """Save a vtk representation of the curve."""
        self.get_pyvista_polyline().save(path)
