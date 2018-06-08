# import python modules
import numpy as np
import math


class Rotation(object):
    """
    A class that represents a rotation of a coordinate system.
    The internal parameters are the unit vector n and the rotation angle phi.
    """
    
    # tolerance for zero check
    eps = 1e-10
    
    def __init__(self, *args):
        """
        The default constructor is from an rotation vector n and an angle phi.
        """
        
        self.q = np.zeros(4);
        
        if len(args) == 0:
            # identity element
            self.q[0] = 1
        elif len(args) == 1 and len(args[0]) == 4:
            # set from quaternion
            self.q[:] = args[0]
        elif len(args) == 2:
            # set from vector and rotation angle
            vector = args[0]
            phi = args[1]
            norm = np.linalg.norm(vector)
            if norm < self.eps:
                raise ValueError('The rotation axis can not be a zero vector!')
            self.q = np.zeros(4);
            if np.abs(phi) < self.eps:
                self.q[0] = 1
            else:
                self.q[0] = np.cos(0.5*phi)
                self.q[1:] = np.sin(0.5*phi) * \
                    np.array(vector,dtype=np.float) / norm
        else:
            raise ValueError('The given arguments {} are invalid!'.format(args))
    
    
    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Create the object from a rotation matrix.
        """
        
        q = np.zeros(4)
        q[0] = np.sqrt( max( 0, 1 + R[0,0] + R[1,1] + R[2,2] ) ) / 2
        q[1] = np.sqrt( max( 0, 1 + R[0,0] - R[1,1] - R[2,2] ) ) / 2
        q[2] = np.sqrt( max( 0, 1 - R[0,0] + R[1,1] - R[2,2] ) ) / 2
        q[3] = np.sqrt( max( 0, 1 - R[0,0] - R[1,1] + R[2,2] ) ) / 2
        q[1] = math.copysign( q[1] , R[2,1] - R[1,2] )
        q[2] = math.copysign( q[2] , R[0,2] - R[2,0] )
        q[3] = math.copysign( q[3] , R[1,0] - R[0,1] )
        return cls(q)
    
    
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
        Return the rotation matrix for this rotation.
        (Krenk (3.53))
        """
        
        R = (self.q[0]**2 - np.dot(self.q[1:],self.q[1:])) * np.eye(3) + \
            2 * self.q[0]*self.get_q_skew() + \
            2 * ([self.q[1:]] * np.transpose([self.q[1:]]))
        
        return R
    
    
    def get_q_skew(self):
        """
        Return the matrix \skew{n} for this rotation.
        """
        
        N = np.zeros([3,3])
        N[0,1] = -self.q[3]
        N[0,2] =  self.q[2]
        N[1,0] =  self.q[3]
        N[1,2] = -self.q[1]
        N[2,0] = -self.q[2]
        N[2,1] =  self.q[1]
        return N
    
    
    def get_quaternion(self):
        """
        Return the quaternion for this rotation, as tuple.
        """
        
        return np.array(self.q)
    
    
    def get_roation_vector(self):
        """
        Return the rotation vector for this object.
        """
        
        norm = np.linalg.norm(self.q[1:])
        n = self.q[1:] / norm
        phi = 2 * np.arctan2(norm, self.q[0])
        
        return phi*n
        
    
    def __mul__(self, other):
        """
        Add this rotation to another, or apply it on a vector.
        """
        
        # check if the other object is also a rotation
        if isinstance(other, Rotation):
            # get quaternions of the two objects
            p = self.q
            q = other.q
            # add the rotations
            added_rotation = np.zeros(4)
            added_rotation[0] = p[0] * q[0] - np.dot(p[1:],q[1:]) 
            added_rotation[1:] = p[0] * q[1:] + q[0] * p[1:] + \
                np.cross(p[1:], q[1:])
            return Rotation(added_rotation)
        elif isinstance(other, np.ndarray):
            # apply rotation to vector
            return np.dot(self.get_rotation_matrix(), other)
        else:
            print("Error, not implemented, does not make sense anyway!")
    
    
    def __eq__(self, other):
        """
        Check if the other rotation is equal to this one
        """
        
        if isinstance(other, Rotation):
            if (np.linalg.norm(self.q - other.q) < self.eps) or \
                (np.linalg.norm(self.q + other.q) < self.eps):
                return True
            else:
                return False
        else:
            return object.__eq__(self, other)
    
        
    def get_dat(self):
        """
        Return a string with the triad components for the .dat line
        """
        
        rotation_vector = self.get_roation_vector()
        return ' {} {} {}'.format(
            rotation_vector[0],
            rotation_vector[1],
            rotation_vector[2]
            )

    
    def __str__(self):
        """
        String representation of object.
        """
        
        return 'Rotation:\n    q0: {}\n    q: {}'.format(
            str(self.q[0]),
            str(self.q[1:])
            )
