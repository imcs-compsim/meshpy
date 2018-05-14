import unittest
import numpy as np
from rotation import Rotation



class TestRotation(unittest.TestCase):
    """
    Test the rotation class
    """
    
    def test_cartesian_rotations(self):
        """
        Create a unitrotation in all 3 directions
        """
        
        # get the 2D rotation matrix
        theta = np.pi/5
        c, s = np.cos(theta), np.sin(theta)
        rot2D = np.array(((c,-s), (s, c)))
        
        for i in range(3):
            
            # get the 3D matrix from the 2D matrix and compare with
            # rotation object
            index = [np.mod(j,3) for j in range(i,i+3) if not j == i]
            
            rot3D = np.eye(3)
            rot3D[np.ix_(index,index)] = rot2D
            
            axis = np.zeros(3)
            axis[i] = 1
            angle = theta
            R = Rotation(axis, angle).get_rotation_matrix()
            
            self.assertAlmostEqual(np.linalg.norm(rot3D - R), 0.)


if __name__ == '__main__':
    unittest.main()