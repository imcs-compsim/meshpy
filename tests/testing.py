# -*- coding: utf-8 -*-
"""
This script is used to test the functionality of the meshpy module.
"""

# Python imports.
import unittest
import numpy as np
import os
import subprocess
import random
import shutil
import glob

# Meshpy imports.
from meshpy import mpy, Rotation, InputFile, InputSection, Material, Mesh, \
    Function, Beam3rHerm2Lin3, BoundaryCondition, Node


# Define the testing paths.
testing_path = '/home/ivo/dev/meshpy/tests'
testing_input = os.path.join(testing_path, 'input-solid-mesh')
testing_temp = os.path.join(testing_path, 'testing-tmp')
baci_path = '/home/ivo/baci/work/release'
baci_release = os.path.join(baci_path, 'baci-release')


def check_tmp_dir():
    """Check if the temp directory exists, if not create it."""
    if not os.path.exists(testing_temp):
        os.makedirs(testing_temp)




        
class TestRotation(unittest.TestCase):
    """This class tests the implementation of the Rotation class."""
    
    
    def rotation_matrix(self, axis, alpha):
        """
        Create a rotation about one of the cartesian axis.
        
        Args
        ----
        axis: int
            0 - x
            1 - y
            2 - z
        angle: double rotation angle
        
        Return
        ----
        rot3D: array(3x3)
            Rotation matrix for this rotation
        """
        c, s = np.cos(alpha), np.sin(alpha)
        rot2D = np.array(((c,-s), (s, c)))
        index = [np.mod(j,3) for j in range(axis,axis+3) if not j == axis]
        rot3D = np.eye(3)
        rot3D[np.ix_(index,index)] = rot2D
        return rot3D
    
    
    def test_cartesian_rotations(self):
        """
        Create a rotation in all 3 directions. And compare with the rotation
        matrix.
        """
        
        theta = 1.        
        # Loop per directions.
        for i in range(3):
            rot3D = self.rotation_matrix(i, theta)
            axis = np.zeros(3)
            axis[i] = 1
            angle = theta
            rotation = Rotation(axis, angle)
            
            # Check if the rotation is the same if it is created from its own
            # quaternion and then created from its own rotation matrix.
            rotation = Rotation(rotation.get_quaternion())
            rotation_matrix = Rotation.from_rotation_matrix(
                rotation.get_rotation_matrix())
            
            self.assertAlmostEqual(
                np.linalg.norm(rot3D - rotation_matrix.get_rotation_matrix()),
                0.)
    
    
    def test_euler_angles(self):
        """Create a rotation with euler angles and compare to known results."""
        
        # Euler angles.
        alpha = 1.1
        beta = 1.2 * np.pi * 10
        gamma = -2.5
        
        # Create the rotation with rotation matrices. 
        Rx = self.rotation_matrix(0,alpha)
        Ry = self.rotation_matrix(1,beta)
        Rz = self.rotation_matrix(2,gamma)
        R_euler = Rz.dot(Ry.dot(Rx))
        
        # Create the rotation with the Rotation object.
        rotation_x = Rotation([1, 0, 0], alpha)
        rotation_y = Rotation([0, 1, 0], beta)
        rotation_z = Rotation([0, 0, 1], gamma)
        rotation_euler = rotation_z * rotation_y * rotation_x
        self.assertAlmostEqual(
            np.linalg.norm(R_euler - rotation_euler.get_rotation_matrix()), 0.)
        self.assertTrue(
            rotation_euler == Rotation.from_rotation_matrix(R_euler)
            )
        
        # Direct formula for quaternions for Euler angles.
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
        self.assertTrue(
            Rotation(quaternion) == Rotation(rotation_euler.get_quaternion()))
        self.assertTrue(
            Rotation(quaternion) == Rotation.from_rotation_matrix(R_euler))
    
    
    def test_negative_angles(self):
        """
        Check if a rotation is created correctly if a negative angle or a large
        angle is given.
        """

        vector = 10*np.array([-1.234243,-2.334343,-1.123123])
        phi = -12.152101868665
        rot = Rotation(vector, phi)
        for i in range(2):
            self.assertTrue(rot == Rotation(vector, phi + 2*i*np.pi))
        
        q = np.array([0.97862427,-0.0884585,-0.16730294,-0.0804945])
        self.assertTrue(Rotation(q)==Rotation(-q))



def create_test_mesh(mesh):
    """Fill the mesh with a couple of test nodes and elements."""
    
    # Set the seed for the pseudo random numbers
    random.seed(0)
    
    # Add material to mesh.
    material = Material('material', 0.1, 0.1, 0.1, 0.1, shear_correction=1.1)
    mesh.add(material)
    
    # Add three test nodes and add them to a beam element
    for j in range(3):
        mesh.add(Node(
            [100*random.uniform(-1,1) for i in range(3)],
            rotation = Rotation(
                [100*random.uniform(-1,1) for i in range(3)],
                100*random.uniform(-1,1)
                )))
    beam = Beam3rHerm2Lin3(material=material, nodes=mesh.nodes)
    mesh.add(beam)
    
    # Add a beam line with three elements
    mesh.create_beam_mesh_line(Beam3rHerm2Lin3, material,
        [100*random.uniform(-1,1) for i in range(3)],
        [100*random.uniform(-1,1) for i in range(3)],
        n_el=3)
    
 
 
  
class TestMeshpy(unittest.TestCase):
    """Test various stuff from the meshpy module."""
     
    def compare_strings(self, name, string_ref, string_compare):
        """
        Compare two stings. If they are not identical open kompare and show
        differences.
        """
        
        # Check if the strings are equal, if not fail the test and show the
        # differences in the strings.
        compare = string_ref == string_compare
        if not compare:
            check_tmp_dir()
            strings = [string_ref, string_compare]
            files = []
            files.append(os.path.join(testing_temp, '{}_ref.dat'.format(name)))
            files.append(
                os.path.join(testing_temp, '{}_compare.dat'.format(name)))
            for i, file in enumerate(files):
                with open(file, 'w') as input_file:
                    input_file.write(strings[i])
            child = subprocess.Popen(
                ['kompare', files[0], files[1]], stderr=subprocess.PIPE)
            child.communicate()
        self.assertTrue(compare, name)
     
     
    def test_mesh_rotations(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self.
        """
        
        mesh_1 = InputFile()
        create_test_mesh(mesh_1)
         
        mesh_2 = InputFile()
        create_test_mesh(mesh_2)
    
        # Set the seed for the pseudo random numbers
        random.seed(0)
        rot = Rotation(
            [100*random.uniform(-1,1) for i in range(3)],
            100*random.uniform(-1,1)
            )
        origin = [100*random.uniform(-1,1) for i in range(3)]

        for node in mesh_1.nodes:
            node.rotate(rot, origin=origin)
         
        mesh_2.rotate(rot, origin=origin)
         
        check_tmp_dir()
        string1 = mesh_1.get_string(header=False)
        string2 = mesh_2.get_string(header=False)
        self.compare_strings('Rotate_mesh', string1, string2)

 
    def test_mesh_rotations_individual(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self, when an array is passed with different rotations.
        """
        
        mesh_1 = InputFile()
        create_test_mesh(mesh_1)
         
        mesh_2 = InputFile()
        create_test_mesh(mesh_2)
    
        # Set the seed for the pseudo random numbers
        random.seed(0)
        
        # Rotate each node with a different rotation
        rotations = np.zeros([len(mesh_1.nodes),4])
        origin = [100*random.uniform(-1,1) for i in range(3)]
        for j, node in enumerate(mesh_1.nodes):
            rot = Rotation(
                [100*random.uniform(-1,1) for i in range(3)],
                100*random.uniform(-1,1)
                )
            rotations[j,:] = rot.get_quaternion()
            node.rotate(rot, origin=origin)

        mesh_2.rotate(rotations, origin=origin)
         
        check_tmp_dir()
        string1 = mesh_1.get_string(header=False)
        string2 = mesh_2.get_string(header=False)
        self.compare_strings('Rotate_mesh_individual', string1, string2)

 
          
  
class TestFullBaci(unittest.TestCase):
    """
    Create and run input files in Baci. They are test files and Baci should
    return 0.
    """
           
    def run_baci_test(self, name, mesh, n_proc=2):
        """
        Run Baci with a input file and check the output. If the test passes,
        the created files are deleted.
        
        Args
        ----
        name: str
            Name of the test case.
        mesh: InputFile
            The InputFile object that contains the simulation.
        n_proc: int
            Number of processors to run Baci on.
        """
        
        # Check if temp directory exists. 
        check_tmp_dir()
        
        # Create input file.
        input_file = os.path.join(testing_temp, name + '.dat')
        mesh.write_input_file(input_file)
        
        # Run Baci with the input file.
        child = subprocess.Popen([
            'mpirun', '-np', str(n_proc),
            baci_release,
            os.path.join(testing_path, input_file),
            os.path.join(testing_temp, 'xxx_' + name)
            ], cwd=testing_temp, stdout=subprocess.PIPE)
        child.communicate()[0]
        self.assertEqual(0, child.returncode,
            msg='Test {} failed!'.format(name)
            )
            
        # If successful delete created files directory.
        if int(child.returncode) == 0:
            os.remove(input_file)
            items = glob.glob(testing_temp+'/xxx_' + name + '*')
            for item in items:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)

       
    def test_honeycomb_as_input(self):
        """
        Create the same honeycomb mesh as defined in 
        /Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_\
        honeycomb_stretch_r01_circ10.dat
        The honeycomb beam is in contact with a rigid sphere, the sphere is
        moved compared to the original test file, since there are some problems
        with the contact convergence. The sphere is imported as an existing
        mesh. 
        """
            
        # Read input file with information of the sphere and simulation.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='honeycomb beam in contact with sphere',
            dat_file=os.path.join(testing_input, 'honeycomb-sphere.dat')
            )
        
        # Modify the time step options.
        input_file.add(InputSection(
            'STRUCTURAL DYNAMIC',
            'NUMSTEP 5',
            'TIMESTEP 0.2',
            option_overwrite=True
            ))
            
        # First delete the results given in the input file and then add the
        # correct results to the file.
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
            
        # Material for the beam.
        material = Material(
            'MAT_BeamReissnerElastHyper',
            2.07e2, # E-Modul
            0, # nu
            1e-3, # rho
            0.2, # diameter of beam
            shear_correction=1.1
            )
            
        # Create the honeycomb mesh.
        mesh_honeycomb = Mesh()
        honeycomb_set = mesh_honeycomb.create_beam_mesh_honeycomb(
            Beam3rHerm2Lin3, material, 50.0, 10, 4, 1, closed_top=False)
        mesh_honeycomb.rotate(Rotation([0,0,1], np.pi/2))
            
        # Functions for the boundary conditions
        ft = Function(
            'COMPONENT 0 FUNCTION a\n' + \
            'VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 ' + \
            'TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0'
            )
        mesh_honeycomb.add(ft)
        
        # Change the sets to lines, only for purpose of matching the test file
        honeycomb_set['bottom'].geo_type = mpy.line
        honeycomb_set['top'].geo_type = mpy.line
        mesh_honeycomb.add(
            BoundaryCondition(honeycomb_set['bottom'],
                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 ' + \
                'FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.dirichlet
            ))
        mesh_honeycomb.add(
            BoundaryCondition(honeycomb_set['top'],
                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 ' + \
                'FUNCT 0 0 {} 0 0 0 0 0 0',
                format_replacement=[ft],
                bc_type=mpy.dirichlet
            ))
            
        # Add the mesh to the imported solid mesh.
        input_file.add(mesh_honeycomb)
        
        # Run the input file in Baci.
        self.run_baci_test('honeycomb-sphere', input_file)
        
        
    def test_beam_and_solid_tube(self):
        """Merge a solid tube with a beam tube and simulate them together."""
        
        # Create the input file and read solid mesh data.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Solid tube with beam tube')
        input_file.read_dat(os.path.join(testing_input, 'tube.dat'))
        
        # Add options for beam_output.
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT/BEAMS',
            '''
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            '''))
        
        # Add functions for boundary conditions and material.
        sin = Function('COMPONENT 0 FUNCTION sin(t*2*pi)')
        cos = Function('COMPONENT 0 FUNCTION cos(t*2*pi)')
        material = Material('MAT_BeamReissnerElastHyper', 1e9, 0, 1e-3, 0.5,
            shear_correction=0.75)
        input_file.add(sin, cos, material)
        
        # Add a straight beam.
        input_file.add(material)
        cantilever_set = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, 
            material, [2,0,-5], [2,0,5], 3)
            
        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(
                cantilever_set['start'], # bc set
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 ' + \
                'FUNCT 0 0 0 0 0 0 0 0 0', # bc string
                bc_type=mpy.dirichlet
                )
            )
        input_file.add(
            BoundaryCondition(
                cantilever_set['end'], # bc set
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 ' + \
                'FUNCT {} {} 0 0 0 0 0 0 0', # bc string
                format_replacement=[cos,sin],
                bc_type = mpy.dirichlet
                )
            )
            
        # Add results for the simulation.
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
        
        # Run the input file in Baci.
        self.run_baci_test('tube', input_file)
           
           
    def test_honeycomb_variants(self):
        """
        Create a few different honeycomb structures.
        """
           
        # Create input file.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Varieties of honeycomb'
            )
        
        # Set options with different syntaxes.
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
        input_file.add('''
            ------------------------------------STRUCTURAL DYNAMIC
            LINEAR_SOLVER                         1
            INT_STRATEGY                          Standard
            DYNAMICTYP                            Statics
            RESULTSEVRY                           1
            NLNSOL                                fullnewton
            PREDICT                               TangDis
            TIMESTEP                              1.
            NUMSTEP                               666
            MAXTIME                               1.0
            TOLRES                                1.0E-4
            TOLDISP                               1.0E-11
            NORM_RESF                             Abs
            NORM_DISP                             Abs
            NORMCOMBI_RESFDISP                    And
            MAXITER                               20
            ''')
        input_file.add(InputSection('STRUCTURAL DYNAMIC', 'NUMSTEP 10',
            option_overwrite=True))
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
           
        # Create four meshes with different types of honeycomb structure.
        mesh = Mesh()
        material = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0, 1e-3, 0.2,
            shear_correction=1.1)
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
                honeycomb_set = mesh.create_beam_mesh_honeycomb(
                    Beam3rHerm2Lin3, material, 10, 6, 3, n_el=2,
                    vertical=vertical, closed_top=closed_top)
                mesh.add(
                        BoundaryCondition(honeycomb_set['bottom'],
                           'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL ' + \
                           '0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                           bc_type=mpy.dirichlet
                        ))
                mesh.add(
                        BoundaryCondition(honeycomb_set['top'],
                           'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL ' + \
                           '1. 1. 1. 0 0 0 0 0 0 FUNCT {0} {0} {0} 0 0 0 0 0 0',
                           format_replacement=[ft[counter]],
                           bc_type=mpy.dirichlet
                        ))
                counter += 1
        
        # Add to input file and set testing results.
        input_file.add(mesh)
        input_file.add(InputSection(
            'RESULT DESCRIPTION',
            '''
            STRUCTURE DIS structure NODE 190 QUANTITY dispx VALUE 2.22755241743985061e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 190 QUANTITY dispy VALUE 7.50563273570252321e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 190 QUANTITY dispz VALUE 2.98689476623922590e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 470 QUANTITY dispx VALUE 2.38369093851331260e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 470 QUANTITY dispy VALUE 7.77848541056979759e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 470 QUANTITY dispz VALUE 3.02586157057907812e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 711 QUANTITY dispx VALUE 0.24316547992973625 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 711 QUANTITY dispy VALUE 0.80121852043307218 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 711 QUANTITY dispz VALUE 0.46918376976622778 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1071 QUANTITY dispx VALUE 0.32535024034244314 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1071 QUANTITY dispy VALUE 1.0426432941382124 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1071 QUANTITY dispz VALUE 0.54691921102400531 TOLERANCE 1e-10
            '''
            ))
           
        # Run the input file in Baci.
        self.run_baci_test('honeycomb-variants', input_file)

if __name__ == '__main__':
    unittest.main()
    pass
