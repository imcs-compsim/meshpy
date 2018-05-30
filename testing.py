import unittest
import numpy as np
import os
import subprocess

# import modules from meshgen
from meshgen.rotation import Rotation
from meshgen.inputfile import InputFile, InputSection
from meshgen.mesh import Material, Mesh, Function, Beam3rHerm2Lin3, ContainerGeom, __LINE__, GeometrySet, BC

# global variables
__testing_path__ = '/home/ivo/dev/inputgenerator-py/tests'
__baci_path__ = '/home/ivo/baci/work/release/baci-release'

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


class TestInputFile(unittest.TestCase):
    """
    Test the input files created.
    """
    
    def run_baci_test(self, input_file, n_proc=2):
        """
        Run baci with a testinput and return the output.
        """
        
        return_val = subprocess.call([
            'mpirun', '-np', str(n_proc),
            __baci_path__,
            os.path.join(__testing_path__, input_file),
            os.path.join(__testing_path__, 'testing-runs', 'xxx')
            ], stdout=subprocess.DEVNULL)
        
        self.assertEqual(0, return_val, msg=input_file)
        
    
        
    def test_honeycomb_as_input(self):
        """
        Create the same honeycomb mesh as defined in 
        /Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_honeycomb_stretch_r01_circ10.dat
        The honeycomb beam is in contact with a rigid sphere, the sphere is moved compared
        to the original test file, since there are some problems with the contact convergence.
        The sphere is imported as an existing mesh. 
        """
        
        # read input file with information on sphere
        input_file = InputFile(maintainer='Ivo Steinbrecher', description='honeycomb beam in contact with sphere')
        input_file.read_dat('/home/ivo/dev/inputgenerator-py/tests/input-solid-mesh/honeycomb-sphere.dat')
        
        # overwrite some entries in the input file
        input_file.add_section(InputSection(
            'STRUCTURAL DYNAMIC',
            '''
            NUMSTEP                         40
            ''',
            option_overwrite=True))
        
        # add results to the input file
        input_file.add_section(InputSection(
            'RESULT DESCRIPTION',
            '''
            STRUCTURE DIS structure NODE 268 QUANTITY dispx VALUE  0.00000000000000000e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 268 QUANTITY dispy VALUE -8.09347205557697258e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 268 QUANTITY dispz VALUE  2.89298034569662965e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 188 QUANTITY dispx VALUE  0.00000000000000000e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 188 QUANTITY dispy VALUE -8.09347205557697258e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 188 QUANTITY dispz VALUE  2.89298034569662965e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 182 QUANTITY dispx VALUE  0.00000000000000000e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 182 QUANTITY dispy VALUE -8.09347205557697258e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 182 QUANTITY dispz VALUE  2.89298034569662965e+00 TOLERANCE 1e-10
            '''
            ))
        
        # material for the beam
        material = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0, 1e-3, 0.2, shear_correction=1.1)
        
        # create the honeycomb mesh
        mesh_honeycomb = Mesh(name='honeycomb_' + str(1))
        honeycomb_set = mesh_honeycomb.add_beam_mesh_honeycomb(Beam3rHerm2Lin3, material,
                               50,
                               10,
                               4,
                               1)
        mesh_honeycomb.rotate(Rotation([0,0,1], np.pi/2))
        
        # define functions for the bc
        ft = Function('COMPONENT 0 FUNCTION a\nVARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0')
        mesh_honeycomb.add_function(ft)
            
        # add bcs
        bc_set = ContainerGeom()
        bc_set.append_item(__LINE__, GeometrySet('line', nodes = honeycomb_set.point[0].nodes))
        bc_set.append_item(__LINE__, GeometrySet('line2', nodes = honeycomb_set.point[1].nodes))
        mesh_honeycomb.sets.merge_containers(bc_set)
      
        mesh_honeycomb.add_bc('dirich',
                BC(bc_set.line[0],
                   'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0'
                ))
        mesh_honeycomb.add_bc('dirich',
                BC(bc_set.line[1],
                   'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0',
                   format_replacement=[ft]
                ))
        
        # add the beam mesh to the solid mesh
        input_file.add_mesh(mesh_honeycomb)
            
        # write input file
        input_dat_file = os.path.join(__testing_path__, 'testing-input/honeycomb-sphere.dat')
        input_file.write_input_file(input_dat_file, print_set_names=False, print_all_sets=False)
        
        # test input
        self.run_baci_test(input_dat_file)


if __name__ == '__main__':
    unittest.main()
