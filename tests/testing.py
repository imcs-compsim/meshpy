import unittest
import numpy as np
import os
import subprocess
import shutil
import filecmp

# import modules from meshgen
from meshpy import Rotation, InputFile, InputSection, Material, Mesh, \
    Function, Beam3rHerm2Lin3, BC, Node, mpy
from meshpy.geometry_set import NodeSetContainer

# global variables
__testing_path__ = '/home/ivo/dev/meshpy/tests'
__testing_input__ = os.path.join(__testing_path__, 'input-solid-mesh')
__testing_temp__ = os.path.join(__testing_path__, 'testing-tmp')
__baci_path__ = '/home/ivo/baci/work/release'
__baci_release__ = os.path.join(__baci_path__, 'baci-release')


def check_tmp_dir():
    """
    Check if the temp directory exists. If not create it.
    """
    
    if not os.path.exists(__testing_temp__):
        os.makedirs(__testing_temp__)


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
            quaternion = Rotation(rotation.get_quaternion())
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
        self.assertTrue(Rotation(quaternion) == rotation_euler)
        self.assertTrue(Rotation(quaternion) == Rotation(rotation_euler.get_quaternion()))
        self.assertTrue(Rotation(quaternion) == Rotation.from_rotation_matrix(R_euler))
    
    
    def test_negative_angles(self):
        """
        Check if a rotation is created correctly if a negative angle or a large
        angle is given.
        """

        vector = np.array([-1.234243,-2.334343,-1.123123])
        phi = -12.152101868665
        rot = Rotation(vector, phi)
        for i in range(2):
            self.assertTrue(rot == Rotation(vector, phi + 2*i*np.pi))
        
        q = np.array([0.97862427,-0.0884585,-0.16730294,-0.0804945])
        self.assertTrue(Rotation(q)==Rotation(-q))
            


def create_test_mesh(mesh):
    """
    Fill the mesh with a couple of test nodes
    """
    
    mesh.add(Node([1.2323342333,2.2314234,4.12313123123], rotation=Rotation([1,2,3],np.pi)))
    mesh.add(Node([123.23342333,22.34234,0.4512313123123], rotation=Rotation([-1,4,-3],3.54545*np.pi)))
    mesh.add(Node([1, 2, 3], rotation=Rotation([1,1,1],2*np.pi)))
    mat = Material('tmp', 1, 0.3, 0.1, 0.2, shear_correction=1.1)    
    tmp_beam = Beam3rHerm2Lin3(material=mat, nodes=mesh.nodes)
    mesh.add(tmp_beam)
    mesh.add(mat)


 
class TestMeshpy(unittest.TestCase):
    """
    Test various stuff from the meshpy module.
    """
    
    def compare_files(self, file1, file2):
        """
        Compare two files. If they are not identical open kompare and show
        differences.
        """
        
        compare = filecmp.cmp(file1, file2, shallow=False)
        if not compare:
            child = subprocess.Popen(['kompare', file1, file2], stderr=subprocess.PIPE)
            child.communicate()
        return compare
    
    
    def test_mesh_rotations(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self. 
        """
         
        mesh_ref = InputFile()
        create_test_mesh(mesh_ref)
        
        mesh_2 = InputFile()
        create_test_mesh(mesh_2)
        
        rot = Rotation([1.1213,-12.2323,-0.123123],1.123123)
        origin = [.1221, -112.11212, 12.12112]
        
        for node in mesh_ref.nodes:
            node.rotate(rot, origin=origin)
        
        mesh_2.rotate(rot, origin=origin)
        
        check_tmp_dir()
        file1 = os.path.join(__testing_temp__, 'mesh_ref.dat')
        file2 = os.path.join(__testing_temp__, 'mesh_2.dat')
        mesh_ref.write_input_file(file1, header=False)
        mesh_2.write_input_file(file2, header=False)
        self.assertTrue(
            self.compare_files(file1, file2),
            'Compare rotation node-wise and mesh-wise'
            )
        

class TestFullBaci(unittest.TestCase):
    """
    Test the input files created with baci.
    """
     
    def run_baci_test(self, input_file, n_proc=2):
        """
        Run baci with a testinput and return the output.
        """
         
        test_name = os.path.splitext(os.path.basename(input_file))[0]
        child = subprocess.Popen([
            'mpirun', '-np', str(n_proc),
            __baci_release__,
            os.path.join(__testing_path__, input_file),
            os.path.join(__testing_temp__, 'xxx_' + test_name)
            ], cwd=__testing_temp__, stdout=subprocess.PIPE)
        child.communicate()[0]
        self.assertEqual(0, child.returncode,
            msg='Test {} failed!'.format(test_name)
            )
         
        # if successful delete tmp directory
        if int(child.returncode) == 0:
            shutil.rmtree(__testing_temp__)
         
     
    def test_honeycomb_as_input(self):
        """
        Create the same honeycomb mesh as defined in 
        /Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_honeycomb_stretch_r01_circ10.dat
        The honeycomb beam is in contact with a rigid sphere, the sphere is moved compared
        to the original test file, since there are some problems with the contact convergence.
        The sphere is imported as an existing mesh. 
        """
          
        # read input file with information on sphere
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='honeycomb beam in contact with sphere',
            dat_file=os.path.join(__testing_input__, 'honeycomb-sphere.dat')
            )
          
        # overwrite some entries in the input file
        input_file.add(InputSection(
            'STRUCTURAL DYNAMIC',
            'NUMSTEP 40',
            option_overwrite=True
            ))
          
        # add results to the input file
        input_file.delete_section('RESULT DESCRIPTION')
        input_file.add(InputSection(
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
        material = Material(
            'MAT_BeamReissnerElastHyper',
            2.07e2, # E-Modul
            0, # nu
            1e-3, # rho
            0.2, # diameter of beam
            shear_correction=1.1
            )
          
        # create the honeycomb mesh
        mesh_honeycomb = Mesh(name='honeycomb_' + str(1))
        honeycomb_set = mesh_honeycomb.add_beam_mesh_honeycomb(
            Beam3rHerm2Lin3,
            material,
            50.0,
            10,
            4,
            1,
            closed_top=False
            )
        mesh_honeycomb.rotate(Rotation([0,0,1], np.pi/2))
          
        # define functions for the bc
        ft = Function(
            'COMPONENT 0 FUNCTION a\n' + \
            'VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 ' + \
            'TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0'
            )
        mesh_honeycomb.add(ft)
        
        # change the sets to lines, only for purpose of matching the test file
        honeycomb_set['bottom'].geo_type = mpy.line
        honeycomb_set['top'].geo_type = mpy.line
        mesh_honeycomb.add(
                BC(honeycomb_set['bottom'],
                   'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                   bc_type=mpy.dirichlet
                ))
        mesh_honeycomb.add(
                BC(honeycomb_set['top'],
                   'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0',
                   format_replacement=[ft],
                   bc_type=mpy.dirichlet
                ))
          
        # add the beam mesh to the solid mesh
        input_file.add(mesh_honeycomb)
              
        # write input file
        check_tmp_dir()
        input_dat_file = os.path.join(__testing_temp__, 'honeycomb-sphere.dat')
        input_file.write_input_file(input_dat_file, print_set_names=False, print_all_sets=False)
          
        # test input
        self.run_baci_test(input_dat_file)
      
      
    def test_beam_and_solid_tube(self):
        """
        Create a solid mesh with cubit and insert some beams into the input file.
        """
      
        # create input file
        input_file = InputFile(maintainer='Ivo Steinbrecher', description='Solid tube with beam tube')
          
        # load solid mesh
        input_file.read_dat(os.path.join(__testing_input__, 'tube.dat'))
          
        # delete solver 2 section
        input_file.delete_section('TITLE')
          
        # add options for beam_output
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT/BEAMS',
            '''
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            '''))
          
        # add a straight line beam
        material = Material('MAT_BeamReissnerElastHyper', 1e9, 0, 1e-3, 0.5)
        cantilever = Mesh(name='cantilever')
        cantilever_set = cantilever.add_beam_mesh_line(Beam3rHerm2Lin3, material, [2,0,-5], [2,0,5], 3)
          
        # add fix at start of the beam
        cantilever.add(
            BC(
                cantilever_set['start'], # bc set
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0' # bc string
                ,bc_type=mpy.dirichlet
                )
            )
          
        # add displacement controlled bc at end of the beam
        sin = Function('COMPONENT 0 FUNCTION sin(t*2*pi)')
        cos = Function('COMPONENT 0 FUNCTION cos(t*2*pi)')
        cantilever.add(sin, cos)
        cantilever.add(
            BC(
                cantilever_set['end'], # bc set
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 FUNCT {} {} 0 0 0 0 0 0 0', # bc string
                format_replacement=[cos,sin],
                bc_type = mpy.dirichlet
                )
            )
          
        # add the beam mesh to the solid mesh
        input_file.add(cantilever)
          
        # add test case result description
        input_file.add(InputSection(
            'RESULT DESCRIPTION',
            '''
            STRUCTURE DIS structure NODE 35 QUANTITY dispx VALUE 1.50796091342925e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 35 QUANTITY dispy VALUE 1.31453288915877e-08 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 35 QUANTITY dispz VALUE 0.0439008100184687e+00 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 69 QUANTITY dispx VALUE 0.921450108160878 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 69 QUANTITY dispy VALUE 1.41113401669104e-15 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 69 QUANTITY dispz VALUE 0.0178350143764099 TOLERANCE 1e-10
            '''))
              
        # write input file
        check_tmp_dir()
        input_dat_file = os.path.join(__testing_temp__, 'tube.dat')
        input_file.write_input_file(input_dat_file)
          
        # test input
        self.run_baci_test(input_dat_file)
         
         
    def test_honeycomb_variants(self):
        """
        Create a few different honeycomb structures.
        """
         
        # create input file
        input_file = InputFile(
            maintainer='Ivo Steinbrecher', description='Varieties of honeycomb')
         
        input_file.add(InputSection('PROBLEM SIZE', 'DIM 3'))
        input_file.add('''
        ------------------------------------PROBLEM TYP
        PROBLEMTYP                            Structure
        RESTART                               0
        --------------------------------------IO
        OUTPUT_BIN                            No
        STRUCT_DISP                           No
        FILESTEPS                             1000
        VERBOSITY                             Standard
        ''')
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT',
            '''
            OUTPUT_DATA_FORMAT                    binary
            INTERVAL_STEPS                        1
            EVERY_ITERATION                       No
            '''))
        input_file.add(InputSection(
            'STRUCTURAL DYNAMIC',
            '''
            LINEAR_SOLVER                         1
            INT_STRATEGY                          Standard
            DYNAMICTYP                            Statics
            RESULTSEVRY                           1
            NLNSOL                                fullnewton
            PREDICT                               TangDis
            TIMESTEP                              1.
            NUMSTEP                               10
            MAXTIME                               1.0
            TOLRES                                1.0E-4
            TOLDISP                               1.0E-11
            NORM_RESF                             Abs
            NORM_DISP                             Abs
            NORMCOMBI_RESFDISP                    And
            MAXITER                               20
            '''))
        input_file.add(InputSection(
            'SOLVER 1',
            '''
            NAME                                  Structure_Solver
            SOLVER                                UMFPACK
            '''))
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT/BEAMS',
            '''
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            '''))
         
        # create two meshes with honeycomb structure
        mesh = Mesh(name='mesh')
        material = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0, 1e-3, 0.2, shear_correction=1.1)
        ft = []
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        mesh.add(ft)
         
        counter = 0
        for vertical in [False, True]:
            for closed_top in [False, True]:
                mesh.translate(17 * np.array([1,0,0]))
                honeycomb_set = mesh.add_beam_mesh_honeycomb(
                    Beam3rHerm2Lin3,
                    material,
                    10,
                    6,
                    3,
                    n_el=2,
                    vertical=vertical,
                    closed_top=closed_top
                    )
                mesh.add(
                        BC(honeycomb_set['bottom'],
                           'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                           bc_type=mpy.dirichlet
                        ))
                mesh.add(
                        BC(honeycomb_set['top'],
                           'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 1. 1. 1. 0 0 0 0 0 0 FUNCT {0} {0} {0} 0 0 0 0 0 0',
                           format_replacement=[ft[counter]],
                           bc_type=mpy.dirichlet
                        ))
                counter += 1
         
        # add the beam mesh to the solid mesh
        input_file.add(mesh)
         
        # add results
        input_file.add(InputSection(
            'RESULT DESCRIPTION',
            '''
            STRUCTURE DIS structure NODE 206 QUANTITY dispx VALUE 0.25164402121731655 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 206 QUANTITY dispy VALUE 0.77988990482787646 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 206 QUANTITY dispz VALUE 0.28881703074847209 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 531 QUANTITY dispx VALUE 0.26731807700633681 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 531 QUANTITY dispy VALUE 0.81334961503175585 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 531 QUANTITY dispz VALUE 0.29489958449574871 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 811 QUANTITY dispx VALUE 0.24316547992973625 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 811 QUANTITY dispy VALUE 0.80121852043307218 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 811 QUANTITY dispz VALUE 0.46918376976622778 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1171 QUANTITY dispx VALUE 0.32535024034244314 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1171 QUANTITY dispy VALUE 1.0426432941382124 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1171 QUANTITY dispz VALUE 0.54691921102400531 TOLERANCE 1e-10
            '''
            ))
         
        # write input file
        check_tmp_dir()
        input_dat_file = os.path.join(__testing_temp__, 'honeycomb-variants.dat')
        input_file.write_input_file(input_dat_file)
         
        # test input
        self.run_baci_test(input_dat_file)


if __name__ == '__main__':
    unittest.main()
    pass
