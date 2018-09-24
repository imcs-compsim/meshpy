# -*- coding: utf-8 -*-
"""
This module defines a class that represents a rotation in 3D.
"""

# Python modules.
import numpy as np
from . import mpy


class Rotation(object):
    """
    A class that represents a rotation of a coordinate system.
    The internal parameters are the unit vector n and the rotation angle phi.
    """

    def __init__(self, *args):
        """
        The default constructor is from an rotation vector n and an angle phi.
        """

        self.q = np.zeros(4)

        if len(args) == 0:
            # Identity element.
            self.q[0] = 1
        elif len(args) == 1 and len(args[0]) == 4:
            # Set from quaternion.
            self.q[:] = np.array(args[0])
        elif len(args) == 2:
            # Set from vector and rotation angle.
            vector = args[0]
            phi = args[1]
            norm = np.linalg.norm(vector)
            if np.abs(phi) < mpy.eps_quaternion:
                self.q[0] = 1
            elif norm < mpy.eps_quaternion:
                raise ValueError('The rotation axis can not be a zero vector!')
            else:
                self.q[0] = np.cos(0.5 * phi)
                self.q[1:] = np.sin(0.5 * phi) * \
                    np.array(vector) / norm
        else:
            raise ValueError(
                'The given arguments {} are invalid!'.format(args))

    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Create the object from a rotation matrix.
        """

        q = np.zeros(4)
        q[0] = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        q[1] = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q[2] = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        q[3] = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        q[1] = np.copysign(q[1], R[2, 1] - R[1, 2])
        q[2] = np.copysign(q[2], R[0, 2] - R[2, 0])
        q[3] = np.copysign(q[3], R[1, 0] - R[0, 1])
        return cls(q)

    @classmethod
    def from_basis(cls, t1, t2):
        """
        Create the object from two basis vectors t1, t2.
        t2 will be orthogonalized on t1, and t3 will be calculated with the
        cross product.
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

    def check_uniqueness(self):
        """
        We always want q0 to be positive -> the range for the rotational angle
        is [-pi, pi].
        """

        if self.q[0] < 0:
            self.q = -self.q

    def check_quaternion_constraint(self):
        """We want to check that q.q = 1."""

        if np.abs(1 - np.linalg.norm(self.q)) > mpy.eps_quaternion:
            raise ValueError('The rotation object is corrupted. q.q does not '
                + 'equal 1!\n{}'.format(self))

    def get_rotation_matrix(self):
        """
        Return the rotation matrix for this rotation.
        (Krenk (3.50))
        """

        R = (self.q[0]**2 - np.dot(self.q[1:], self.q[1:])) * np.eye(3) + \
            2 * self.q[0] * self.get_q_skew() + \
            2 * ([self.q[1:]] * np.transpose([self.q[1:]]))

        return R

    def get_q_skew(self):
        r"""
        Return the matrix \skew{n} for this rotation.
        """

        N = np.zeros([3, 3])
        N[0, 1] = -self.q[3]
        N[0, 2] = self.q[2]
        N[1, 0] = self.q[3]
        N[1, 2] = -self.q[1]
        N[2, 0] = -self.q[2]
        N[2, 1] = self.q[1]
        return N

    def get_quaternion(self):
        """
        Return the quaternion for this rotation, as tuple.
        """

        return np.array(self.q)

    def get_rotation_vector(self):
        """
        Return the rotation vector for this object.
        """

        self.check_uniqueness()
        self.check_quaternion_constraint()

        norm = np.linalg.norm(self.q[1:])
        phi = 2 * np.arctan2(norm, self.q[0])

        # Check if effective rotation angle is 0.
        if np.abs(np.sin(phi / 2)) < mpy.eps_quaternion:
            return np.zeros(3)
        else:
            return phi * self.q[1:] / norm

    def inv(self):
        """
        Return the inverse of this rotation.
        """

        tmp_quaternion = self.q.copy()
        tmp_quaternion[0] *= -1.
        return Rotation(tmp_quaternion)

    def __mul__(self, other):
        """
        Add this rotation to another, or apply it on a vector.
        """

        # Check if the other object is also a rotation.
        if isinstance(other, Rotation):
            # Get quaternions of the two objects.
            p = self.q
            q = other.q
            # Add the rotations.
            added_rotation = np.zeros_like(self.q)
            added_rotation[0] = p[0] * q[0] - np.dot(p[1:], q[1:])
            added_rotation[1:] = p[0] * q[1:] + q[0] * p[1:] + \
                np.cross(p[1:], q[1:])
            return Rotation(added_rotation)
        elif (
                (isinstance(other, np.ndarray) or isinstance(other, list))
                and len(other) == 3
                ):
            # Apply rotation to vector.
            return np.dot(self.get_rotation_matrix(), np.array(other))
        elif other is None:
            # If multiplied with none, nothing happens.
            return self
        else:
            print("Error, not implemented, does not make sense anyway!")

    def __eq__(self, other):
        """
        Check if the other rotation is equal to this one
        """

        if isinstance(other, Rotation):
            if (np.linalg.norm(self.q - other.q) < mpy.eps_quaternion) or \
                    (np.linalg.norm(self.q + other.q) < mpy.eps_quaternion):
                return True
            else:
                return False
        else:
            return object.__eq__(self, other)

    def get_dat(self):
        """
        Return a string with the triad components for the .dat line
        """

        rotation_vector = self.get_rotation_vector()
        rotation_string = ' '.join([mpy.dat_precision for _i in range(3)])
        # The zeros are added to avoid negative zeros in the input file.
        return rotation_string.format(
            rotation_vector[0] + 0,
            rotation_vector[1] + 0,
            rotation_vector[2] + 0
            )

    def copy(self):
        """Return a deep copy of this object."""
        return Rotation(self.q)

    def __str__(self):
        """
        String representation of object.
        """

        return 'Rotation:\n    q0: {}\n    q: {}'.format(
            str(self.q[0]),
            str(self.q[1:])
            )


def get_relative_rotation(rotation1, rotation2):
    """Return the rotation from rotation1 to rotation2."""
    return rotation2 * rotation1.inv()
