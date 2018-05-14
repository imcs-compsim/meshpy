import unittest
import numpy as np

# import modules from beamgen
from beamgen.rotation import Rotation


def roation_matrix(axis, alpha):
    """
    Create a roation about an axis
        0 - x
        1 - y
        2 - z
    with the angle alpha.
    """

    c, s = np.cos(alpha), np.sin(alpha)
    rot2D = np.array(((c,-s), (s, c)))
    
    index = [np.mod(j,3) for j in range(axis,axis+3) if not j == axis]
    
    rot3D = np.eye(3)
    rot3D[np.ix_(index,index)] = rot2D
    return rot3D


def quaternion_diff_norm(quaternion1, quaternion2):
    """
    Simple norm of the difference of two quaternions.
    """
    
    if not len(quaternion1) == 4:
        tmp1 = np.zeros(4)
        tmp1[0] = quaternion1[0]
        tmp1[1:] = quaternion1[1]
    else:
        tmp1 = np.array(quaternion1)
    if not len(quaternion2) == 4:
        tmp2 = np.zeros(4)
        tmp2[0] = quaternion2[0]
        tmp2[1:] = quaternion2[1]
    else:
        tmp2 = np.array(quaternion2)
    return np.linalg.norm(tmp1 - tmp2)

        
class TestRotation(unittest.TestCase):
    """
    Test the rotation class.
    """
    
    def test_cartesian_rotations(self):
        """
        Create a unitrotation in all 3 directions.
        """
        
        # angle to rotate
        theta = np.pi/5
        
        for i in range(3):
            
            rot3D = roation_matrix(i, theta)
            
            axis = np.zeros(3)
            axis[i] = 1
            angle = theta
            
            rotation = Rotation(axis, angle)
            quaternion = Rotation.from_quaternion(rotation.get_quaternion())
            rotation_matrix = Rotation.from_rotation_matrix(quaternion.get_rotation_matrix())
            
            self.assertAlmostEqual(np.linalg.norm(rot3D - rotation_matrix.get_rotation_matrix()), 0.)
    
    
    def test_euler_angles(self):
        """
        Create a rotation with euler angles.
        """
        
        # euler angles
        alpha = 1.1
        beta = 1.2
        gamma = 2.5
        
        Rx = roation_matrix(0,alpha)
        Ry = roation_matrix(1,beta)
        Rz = roation_matrix(2,gamma)
        R_euler = Rz.dot(Ry.dot(Rx))
        
        rotation_x = Rotation([1, 0, 0], alpha)
        rotation_y = Rotation([0, 1, 0], beta)
        rotation_z = Rotation([0, 0, 1], gamma)
        rotation_euler = rotation_z * rotation_y * rotation_x
        self.assertAlmostEqual(np.linalg.norm(R_euler - rotation_euler.get_rotation_matrix()), 0.)
        
        # direct formular for quaternions for euler angles
        quaternion = np.zeros(4)
        cy = np.cos(gamma * 0.5);
        sy = np.sin(gamma * 0.5);
        cr = np.cos(alpha * 0.5);
        sr = np.sin(alpha * 0.5);
        cp = np.cos(beta * 0.5);
        sp = np.sin(beta * 0.5);

        quaternion[0] = cy * cr * cp + sy * sr * sp
        quaternion[1] = cy * sr * cp - sy * cr * sp
        quaternion[2] = cy * cr * sp + sy * sr * cp
        quaternion[3] = sy * cr * cp - cy * sr * sp
        self.assertAlmostEqual(quaternion_diff_norm(quaternion, rotation_euler.get_quaternion()), 0.)
        self.assertAlmostEqual(quaternion_diff_norm(
            quaternion,
            Rotation.from_quaternion(rotation_euler.get_quaternion()).get_quaternion()
            ), 0.)
        self.assertAlmostEqual(quaternion_diff_norm(
            quaternion,
            Rotation.from_rotation_matrix(R_euler).get_quaternion()
            ), 0.)


if __name__ == '__main__':
    unittest.main()