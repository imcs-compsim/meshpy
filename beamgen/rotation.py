import numpy as np
import math


class Rotation(object):
    """
    A class that represents a roation of a triad.
    The internal parameters are the unitvector n and the roation angle phi.
    """
    
    def __init__(self, vector, angle, *args, **kwargs):
        """
        The default constructor is from an roation vector n and an angle phi.
        """
        
        # make unitvector
        norm = np.linalg.norm(vector) 
        if not norm == 0:
            self.n = np.array(vector)/norm
            self.phi = angle
        else:
            print("ERROR: rotation vector length is 0!")
    
    
    @classmethod
    def from_quaternion(cls, quaternion):
        """
        Create the object from a quaternion.
        """
        
        cos = quaternion[0]
        sin = np.linalg.norm(quaternion[1])
        
        # check if the rotation angle is not 0
        if np.abs(sin) > 1e-10:
            phi = 2*np.arctan2(sin, cos)
            n = quaternion[1]/sin
        else:
            phi = 0
            n = np.array([1, 0, 0])
        
        return cls(n, phi)
    
    
    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Create the object from a rotation matrix.
        """
        
        q0 = np.sqrt( max( 0, 1 + R[0,0] + R[1,1] + R[2,2] ) ) / 2
        q = np.zeros(3)
        q[0] = np.sqrt( max( 0, 1 + R[0,0] - R[1,1] - R[2,2] ) ) / 2
        q[1] = np.sqrt( max( 0, 1 - R[0,0] + R[1,1] - R[2,2] ) ) / 2
        q[2] = np.sqrt( max( 0, 1 - R[0,0] - R[1,1] + R[2,2] ) ) / 2
        q[0] = math.copysign( q[0] , R[2,1] - R[1,2] )
        q[1] = math.copysign( q[1] , R[0,2] - R[2,0] )
        q[2] = math.copysign( q[2] , R[1,0] - R[0,1] )
        return cls.from_quaternion(
            (q0, q)
            )
    
    
    @classmethod
    def from_basis(cls, t1, t2):
        """
        Create the object from two basis vectors t1, t2.
        t2 will be orthogonalized on t1, and t3 will be calculated with the cross product.
        """
        
        t1_normal = t1 / np.linalg.norm(t1)
        t2_ortho = t2 - t1_normal * np.dot(t1_normal, t2)
        t2_normal = t2_ortho / np.linalg.norm(t2_ortho)
        t3_normal = np.cross(t1_normal, t2_normal)
        
        R = np.transpose([t1_normal, t2_normal, t3_normal])
        return cls.from_rotation_matrix(R)
    
    
    def get_rotation_matrix(self):
        """
        Return the roation matrix for this rotation.
        """
        
        R = np.cos(self.phi) * np.eye(3) + \
            np.sin(self.phi) * self.get_n_skew() + \
            (1 - np.cos(self.phi)) * ([self.n] * np.transpose([self.n]))
        
        return R
    
    
    def get_n_skew(self):
        """
        Return the matrix \skew{n} for this rotation.
        """
        
        N = np.zeros([3,3])
        N[0,1] = -self.n[2]
        N[0,2] =  self.n[1]
        N[1,0] =  self.n[2]
        N[1,2] = -self.n[0]
        N[2,0] = -self.n[1]
        N[2,1] =  self.n[0]
        return N
    
    
    def get_quaternion(self):
        """
        Return the quaternions for this roation, as tupel.
        """
        
        return (
            np.cos(0.5*self.phi),
            np.sin(0.5*self.phi) * self.n
            )
    
    
    def __mul__(self, other):
        """
        Add this roation to another, or apply it on a vector.
        """
        
        # check if the other object is also a roation
        if isinstance(other, Rotation):
            # get quaternions of the two objects
            p = self.get_quaternion()
            q = other.get_quaternion()
            # add the roations
            return Rotation.from_quaternion((
                p[0] * q[0] - np.dot(p[1],q[1]),
                p[0] * q[1] + q[0] * p[1] + np.cross(p[1], q[1])
                ))
        elif isinstance(other, np.ndarray):
            # apply rotation to vector
            return np.dot(self.get_rotation_matrix(), other)
        else:
            print("Error, not implemented, does not make sense anyway!")
    
        
    def get_dat(self):
        """
        Return a string with the triad components for the dat line
        """
        
        rotation_vector = self.n * self.phi
        return ' {} {} {}'.format(
            rotation_vector[0],
            rotation_vector[1],
            rotation_vector[2]
            )

    def __str__(self):
        """
        String representation of object.
        """
        
        return 'Rotation:\n    Vector: {}\n    Angle: {}'.format(
            str(self.n),
            str(self.phi)
            )
