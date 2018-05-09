


import numpy as np



class Rotation(object):
    """
    TODO
    """
    
    def __init__(self, vector, angle, *args, **kwargs):
        """
        The default constructor is from an roation vector n and an angle phi
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
        Create the object from a quaternion
        """
        
        cos = quaternion[0]
        sin = np.linalg.norm(quaternion[1])
        phi = 2*np.arctan2(sin, cos)
        n = quaternion[1]/sin
        
        return cls(n, phi)
    
    
    def get_rotation_matrix(self):
        """
        Return the roation matrix for this rotation
        """
        
        R = np.cos(self.phi) * np.eye(3) + \
            np.sin(self.phi) * self.get_n_skew() + \
            (1 - np.cos(self.phi)) * ([self.n] * np.transpose([self.n]))
        
        return R
    
    
    def get_n_skew(self):
        """
        Return the matrix \skew{n} for this rotation
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
        Return the quaternions for this roation, as tupel
        """
        
        return (
            np.cos(0.5*self.phi),
            np.sin(0.5*self.phi) * self.n
            )
    
    
    def __mul__(self, other):
        """
        Add this roation to another
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










print()
print()
print()




a = Rotation([1, -5, 1], 1)
b = Rotation([1, -5, 1], 1)
d = Rotation([1, -5, 1], 2)
c = a*b

vec = np.array([1, 3, 1])

print()
print(a*(b*vec))
print(a*b*vec)
print(c*vec)
print(d*vec)









print("end")