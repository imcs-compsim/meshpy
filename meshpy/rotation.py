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
"""This module defines a class that represents a rotation in 3D."""

import copy

import numpy as np

from . import mpy


def skew_matrix(vector):
    """Return the skew matrix for the vector."""
    skew = np.zeros([3, 3])
    skew[0, 1] = -vector[2]
    skew[0, 2] = vector[1]
    skew[1, 0] = vector[2]
    skew[1, 2] = -vector[0]
    skew[2, 0] = -vector[1]
    skew[2, 1] = vector[0]
    return skew


class Rotation:
    """A class that represents a rotation of a coordinate system.

    Internally the rotations are stored as quaternions.
    """

    def __init__(self, *args):
        """Initialize the rotation object.

        Args
        ----
        *args:
            - Rotation()
                Create a identity rotation.
            - Rotation(axis, phi)
                Create a rotation around the vector axis with the angle phi.
        """

        self.q = np.zeros(4)

        if len(args) == 0:
            # Identity element.
            self.q[0] = 1
        elif len(args) == 2:
            # Set from vector and rotation angle.
            vector = args[0]
            phi = args[1]
            norm = np.linalg.norm(vector)
            if np.abs(phi) < mpy.eps_quaternion:
                self.q[0] = 1
            elif norm < mpy.eps_quaternion:
                raise ValueError("The rotation axis can not be a zero vector!")
            else:
                self.q[0] = np.cos(0.5 * phi)
                self.q[1:] = np.sin(0.5 * phi) * np.array(vector) / norm
        else:
            raise ValueError(f"The given arguments {args} are invalid!")

    @classmethod
    def from_quaternion(cls, q):
        """Create the object from a quaternion float array (4x1)"""
        rotation = object.__new__(cls)
        rotation.q = np.array(q)
        if (not rotation.q.ndim == 1) or (not len(rotation.q) == 4):
            raise ValueError("Got quaternion array with unexpected dimensions")
        return rotation

    @classmethod
    def from_rotation_matrix(cls, R):
        """Create the object from a rotation matrix.

        The code is based on Spurriers algorithm:
            R. A. Spurrier (1978): “Comment on “Singularity-free extraction of a quaternion from a
            direction-cosine matrix”
        """

        q = np.zeros(4)
        trace = np.trace(R)
        values = [R[i, i] for i in range(3)]
        values.append(trace)
        arg_max = np.argmax(values)
        if arg_max == 3:
            q[0] = np.sqrt(trace + 1) * 0.5
            q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
            q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
            q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
        else:
            i_index = arg_max
            j_index = (i_index + 1) % 3
            k_index = (i_index + 2) % 3
            q_i = np.sqrt(R[i_index, i_index] * 0.5 + (1 - trace) * 0.25)
            q[0] = (R[k_index, j_index] - R[j_index, k_index]) / (4 * q_i)
            q[i_index + 1] = q_i
            q[j_index + 1] = (R[j_index, i_index] + R[i_index, j_index]) / (4 * q_i)
            q[k_index + 1] = (R[k_index, i_index] + R[i_index, k_index]) / (4 * q_i)

        return cls.from_quaternion(q)

    @classmethod
    def from_basis(cls, t1, t2):
        """Create the object from two basis vectors t1, t2.

        t2 will be orthogonalized on t1, and t3 will be calculated with
        the cross product.
        """

        t1_normal = t1 / np.linalg.norm(t1)
        t2_ortho = t2 - t1_normal * np.dot(t1_normal, t2)
        t2_normal = t2_ortho / np.linalg.norm(t2_ortho)
        t3_normal = np.cross(t1_normal, t2_normal)

        R = np.transpose([t1_normal, t2_normal, t3_normal])
        return cls.from_rotation_matrix(R)

    @classmethod
    def from_rotation_vector(cls, rotation_vector):
        """Create the object from a rotation vector."""

        phi = np.linalg.norm(rotation_vector)
        if np.abs(phi) < mpy.eps_quaternion:
            return cls([0, 0, 0], 0)
        else:
            return cls(rotation_vector, phi)

    def check(self):
        """Perform all checks for the rotation."""
        self.check_uniqueness()
        self.check_quaternion_constraint()

    def check_uniqueness(self):
        """We always want q0 to be positive -> the range for the rotational
        angle is [-pi, pi]."""

        if self.q[0] < 0:
            self.q = -self.q

    def check_quaternion_constraint(self):
        """We want to check that q.q = 1."""

        if np.abs(1 - np.linalg.norm(self.q)) > mpy.eps_quaternion:
            raise ValueError(
                f"The rotation object is corrupted. q.q does not equal 1! q={self.q}"
            )

    def get_rotation_matrix(self):
        """Return the rotation matrix for this rotation.

        (Krenk (3.50))
        """
        q_skew = skew_matrix(self.q[1:])
        R = (
            (self.q[0] ** 2 - np.dot(self.q[1:], self.q[1:])) * np.eye(3)
            + 2 * self.q[0] * q_skew
            + 2 * ([self.q[1:]] * np.transpose([self.q[1:]]))
        )

        return R

    def get_quaternion(self):
        """Return the quaternion for this rotation, as tuple."""

        return np.array(self.q)

    def get_rotation_vector(self):
        """Return the rotation vector for this object."""

        self.check()

        norm = np.linalg.norm(self.q[1:])
        phi = 2 * np.arctan2(norm, self.q[0])

        # Check if effective rotation angle is 0.
        if np.abs(np.sin(phi / 2)) < mpy.eps_quaternion:
            return np.zeros(3)
        elif np.abs(np.abs(phi) - np.pi) < mpy.eps_quaternion:
            # For rotations of -pi and pi, numerical issues might occur that the output
            # rotation vector is non-deterministic (correct, but the sign can switch).
            # To avoid this, we scale the rotation axis in such a way, that for a
            # rotation angle of -pi or pi, the first component of the rotation axis
            # that is not 0 is positive.
            rotation_axis = self.q[1:] / norm
            for i_dir in range(3):
                if np.abs(rotation_axis[i_dir]) > mpy.eps_quaternion:
                    if rotation_axis[i_dir] > 0:
                        return phi * rotation_axis
                    else:
                        return -phi * rotation_axis

        else:
            return phi * self.q[1:] / norm

    def get_transformation_matrix(self):
        """Return the transformation matrix for this rotation.

        The transformation matrix maps the (infinitesimal)
        multiplicative rotational increments onto the additive ones.
        """

        omega = self.get_rotation_vector()
        omega_norm = np.linalg.norm(omega)

        # We have to take the inverse of the the rotation angle here, therefore,
        # we have a branch for small angles where the singularity is not present.
        if omega_norm**2 > mpy.eps_quaternion:
            # Taken from Jelenic and Crisfield (1999) Equation (2.5)
            omega_dir = omega / omega_norm
            omega_skew = skew_matrix(omega)
            transformation_matrix = (
                np.outer(omega_dir, omega_dir)
                - 0.5 * omega_skew
                + 0.5
                * omega_norm
                / np.tan(0.5 * omega_norm)
                * (np.identity(3) - np.outer(omega_dir, omega_dir))
            )
        else:
            # This is the constant part of the Taylor series expansion. If this
            # function is used with automatic differentiation, higher order
            # terms have to be added!
            transformation_matrix = np.identity(3)
        return transformation_matrix

    def get_transformation_matrix_inv(self):
        """Return the inverse of the transformation matrix for this rotation.

        The inverse of the transformation matrix maps the
        (infinitesimal) additive rotational increments onto the
        multiplicative ones.
        """

        omega = self.get_rotation_vector()
        omega_norm = np.linalg.norm(omega)

        # We have to take the inverse of the the rotation angle here, therefore,
        # we have a branch for small angles where the singularity is not present.
        if omega_norm**2 > mpy.eps_quaternion:
            # Taken from Jelenic and Crisfield (1999) Equation (2.5)
            omega_dir = omega / omega_norm
            omega_skew = skew_matrix(omega)
            transformation_matrix_inverse = (
                (1.0 - np.sin(omega_norm) / omega_norm) * np.outer(omega_dir, omega_dir)
                + np.sin(omega_norm) / omega_norm * np.identity(3)
                + (1.0 - np.cos(omega_norm)) / omega_norm**2 * omega_skew
            )
        else:
            # This is the constant part of the Taylor series expansion. If this
            # function is used with automatic differentiation, higher order
            # terms have to be added!
            transformation_matrix_inverse = np.identity(3)
        return transformation_matrix_inverse

    def inv(self):
        """Return the inverse of this rotation."""

        tmp_quaternion = self.q.copy()
        tmp_quaternion[0] *= -1.0
        return Rotation.from_quaternion(tmp_quaternion)

    def __mul__(self, other):
        """Add this rotation to another, or apply it on a vector."""

        # Check if the other object is also a rotation.
        if isinstance(other, Rotation):
            # Get quaternions of the two objects.
            p = self.q
            q = other.q
            # Add the rotations.
            added_rotation = np.zeros_like(self.q)
            added_rotation[0] = p[0] * q[0] - np.dot(p[1:], q[1:])
            added_rotation[1:] = p[0] * q[1:] + q[0] * p[1:] + np.cross(p[1:], q[1:])
            return Rotation.from_quaternion(added_rotation)
        elif isinstance(other, (list, np.ndarray)) and len(other) == 3:
            # Apply rotation to vector.
            return np.dot(self.get_rotation_matrix(), np.array(other))
        raise NotImplementedError("Error, not implemented, does not make sense anyway!")

    def __eq__(self, other):
        """Check if the other rotation is equal to this one."""

        if isinstance(other, Rotation):
            return bool(
                (np.linalg.norm(self.q - other.q) < mpy.eps_quaternion)
                or (np.linalg.norm(self.q + other.q) < mpy.eps_quaternion)
            )
        else:
            return object.__eq__(self, other)

    def get_dat(self):
        """Return a string with the triad components for the .dat line."""

        rotation_vector = self.get_rotation_vector()

        # The zeros are added to avoid negative zeros in the input file.
        return " ".join(
            [
                (
                    mpy.dat_precision.format(component + 0)
                    if np.abs(component) >= mpy.eps_quaternion
                    else "0"
                )
                for component in rotation_vector
            ]
        )

    def copy(self):
        """Return a deep copy of this object."""
        return copy.deepcopy(self)

    def __str__(self):
        """String representation of object."""

        self.check()
        return f"Rotation:\n    q0: {self.q[0]}\n    q: {self.q[1:]}"


def get_relative_rotation(rotation1, rotation2):
    """Return the rotation from rotation1 to rotation2."""
    return rotation2 * rotation1.inv()


def add_rotations(rotation_21, rotation_10):
    """Multiply a rotation onto another.

    Args
    ----
    rotation_10: np.ndarray
        Array with the dimensions n x 4 or 4 x 1.
        The first rotation that is applied.
    rotation_21: np.ndarray
        Array with the dimensions n x 4 or 4 x 1.
        The second rotation that is applied.

    Return
    ----
    rot_new: np.ndarray
        Array with the dimensions n x 4.
        This array contains the new quaternions.
    """

    # Transpose the arrays, to work with the following code.
    if isinstance(rotation_10, Rotation):
        rot1 = rotation_10.get_quaternion().transpose()
    else:
        rot1 = np.transpose(rotation_10)
    if isinstance(rotation_21, Rotation):
        rot2 = rotation_21.get_quaternion().transpose()
    else:
        rot2 = np.transpose(rotation_21)

    if rot1.size > rot2.size:
        rotnew = np.zeros_like(rot1)
    else:
        rotnew = np.zeros_like(rot2)

    # Multiply the two rotations (code is taken from /utility/rotation.nb).
    rotnew[0] = (
        rot1[0] * rot2[0] - rot1[1] * rot2[1] - rot1[2] * rot2[2] - rot1[3] * rot2[3]
    )
    rotnew[1] = (
        rot1[1] * rot2[0] + rot1[0] * rot2[1] + rot1[3] * rot2[2] - rot1[2] * rot2[3]
    )
    rotnew[2] = (
        rot1[2] * rot2[0] - rot1[3] * rot2[1] + rot1[0] * rot2[2] + rot1[1] * rot2[3]
    )
    rotnew[3] = (
        rot1[3] * rot2[0] + rot1[2] * rot2[1] - rot1[1] * rot2[2] + rot1[0] * rot2[3]
    )

    return rotnew.transpose()


def rotate_coordinates(coordinates, rotation, *, origin=None):
    """Rotate all given coordinates.

    Args
    ----
    coordinates: np.array
        Array of 3D coordinates to be rotated
    rotation: Rotation, list(quaternions) (nx4)
        The rotation that will be applied to the coordinates. Can also be an
        array with a quaternion for each coordinate.
    origin: 3D vector
        If this is given, the mesh is rotated about this point. Default is
        (0,0,0)
    """

    if isinstance(rotation, Rotation):
        rotation = rotation.get_quaternion().transpose()

    # Check if origin has to be added
    if origin is None:
        origin = [0.0, 0.0, 0.0]

    # New position array
    coordinates_new = np.zeros_like(coordinates)

    # Evaluate the new positions using the numpy data structure
    # (code is taken from /utility/rotation.nb)
    rotation = rotation.transpose()

    q0_q0 = np.square(rotation[0])
    q0_q1_2 = 2.0 * rotation[0] * rotation[1]
    q0_q2_2 = 2.0 * rotation[0] * rotation[2]
    q0_q3_2 = 2.0 * rotation[0] * rotation[3]

    q1_q1 = np.square(rotation[1])
    q1_q2_2 = 2.0 * rotation[1] * rotation[2]
    q1_q3_2 = 2.0 * rotation[1] * rotation[3]

    q2_q2 = np.square(rotation[2])
    q2_q3_2 = 2.0 * rotation[2] * rotation[3]

    q3_q3 = np.square(rotation[3])

    coordinates_new[:, 0] = (
        (q0_q0 + q1_q1 - q2_q2 - q3_q3) * (coordinates[:, 0] - origin[0])
        + (q1_q2_2 - q0_q3_2) * (coordinates[:, 1] - origin[1])
        + (q0_q2_2 + q1_q3_2) * (coordinates[:, 2] - origin[2])
    )
    coordinates_new[:, 1] = (
        (q1_q2_2 + q0_q3_2) * (coordinates[:, 0] - origin[0])
        + (q0_q0 - q1_q1 + q2_q2 - q3_q3) * (coordinates[:, 1] - origin[1])
        + (-q0_q1_2 + q2_q3_2) * (coordinates[:, 2] - origin[2])
    )
    coordinates_new[:, 2] = (
        (-q0_q2_2 + q1_q3_2) * (coordinates[:, 0] - origin[0])
        + (q0_q1_2 + q2_q3_2) * (coordinates[:, 1] - origin[1])
        + (q0_q0 - q1_q1 - q2_q2 + q3_q3) * (coordinates[:, 2] - origin[2])
    )

    if origin is not None:
        coordinates_new += origin

    return coordinates_new


def smallest_rotation(q: Rotation, t):
    """Get the triad that results from the smallest rotation (rotation without
    twist) from the triad q such that the rotated first basis vector aligns
    with t. For more details see Christoph Meier's dissertation chapter 2.1.2.

    Args
    ----
    q: Rotation
        Starting triad.
    t: Vector in R3
        Direction of the first basis of the rotated triad.
    Return
    ----
    q_sr: Rotation
        The triad that results from a smallest rotation.
    """

    R_old = q.get_rotation_matrix()
    g1_old = R_old[:, 0]
    g1 = np.array(t) / np.linalg.norm(t)

    # Quaternion components of relative rotation
    q_rel = np.zeros(4)

    # The scalar quaternion part is cos(alpha/2) this is equal to
    q_rel[0] = np.linalg.norm(0.5 * (g1_old + g1))

    # Vector part of the quaternion is sin(alpha/2)*axis
    q_rel[1:] = np.cross(g1_old, g1) / (2.0 * q_rel[0])

    return Rotation.from_quaternion(q_rel) * q
