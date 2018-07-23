# -*- coding: utf-8 -*-
"""
This script is used to test the functionality of the meshpy module.
"""

# Python imports.
import os
import sys
import unittest
import numpy as np
import autograd.numpy as npAD
import os
import subprocess
import random
import shutil
import glob
import warnings
import vtk

# Set path to find meshpy.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))

# Meshpy imports.
from meshpy import mpy, Rotation, get_relative_rotation, InputFile, \
    InputSection, Material, Mesh, Function, Beam3rHerm2Lin3, \
    BoundaryCondition, Node, BaseMeshItem, VTKWriter, compare_xml


# Define the testing paths.
testing_path = os.path.abspath(os.path.dirname(__file__))
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
        rot2D = np.array(((c, -s), (s, c)))
        index = [np.mod(j, 3) for j in range(axis, axis + 3) if not j == axis]
        rot3D = np.eye(3)
        rot3D[np.ix_(index, index)] = rot2D
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
        Rx = self.rotation_matrix(0, alpha)
        Ry = self.rotation_matrix(1, beta)
        Rz = self.rotation_matrix(2, gamma)
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
        cy = np.cos(gamma * 0.5)
        sy = np.sin(gamma * 0.5)
        cr = np.cos(alpha * 0.5)
        sr = np.sin(alpha * 0.5)
        cp = np.cos(beta * 0.5)
        sp = np.sin(beta * 0.5)
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

        vector = 10 * np.array([-1.234243, -2.334343, -1.123123])
        phi = -12.152101868665
        rot = Rotation(vector, phi)
        for i in range(2):
            self.assertTrue(rot == Rotation(vector, phi + 2 * i * np.pi))

        q = np.array([0.97862427, -0.0884585, -0.16730294, -0.0804945])
        self.assertTrue(Rotation(q) == Rotation(-q))

    def test_inverse_rotation(self):
        """Test the inv() function for rotations."""

        rot = Rotation([1, 2, 3], 2)

        # Check if inverse rotation gets identity rotation. Use two different
        # constructors for identity rotation.
        self.assertTrue(Rotation([0, 0, 0], 0) == rot * rot.inv())
        self.assertTrue(Rotation() == rot * rot.inv())

        # Check that there is no warning or error when getting the vector for
        # an identity rotation.
        (rot * rot.inv()).get_roation_vector()

    def test_relative_roation(self):
        """Test the relative rotation between two rotations."""

        rot1 = Rotation([1, 2, 3], 2)
        rot2 = Rotation([0.1, -0.2, 2], np.pi / 5)

        rot21 = get_relative_rotation(rot1, rot2)

        self.assertTrue(rot2 == rot21 * rot1)


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
            [100 * random.uniform(-1, 1) for i in range(3)],
            rotation=Rotation(
                [100 * random.uniform(-1, 1) for i in range(3)],
                100 * random.uniform(-1, 1)
                )))
    beam = Beam3rHerm2Lin3(material=material, nodes=mesh.nodes)
    mesh.add(beam)

    # Add a beam line with three elements
    mesh.create_beam_mesh_line(Beam3rHerm2Lin3, material,
        [100 * random.uniform(-1, 1) for i in range(3)],
        [100 * random.uniform(-1, 1) for i in range(3)],
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
            [100 * random.uniform(-1, 1) for i in range(3)],
            100 * random.uniform(-1, 1)
            )
        origin = [100 * random.uniform(-1, 1) for i in range(3)]

        for node in mesh_1.nodes:
            node.rotate(rot, origin=origin)

        mesh_2.rotate(rot, origin=origin)

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
        rotations = np.zeros([len(mesh_1.nodes), 4])
        origin = [100 * random.uniform(-1, 1) for i in range(3)]
        for j, node in enumerate(mesh_1.nodes):
            rot = Rotation(
                [100 * random.uniform(-1, 1) for i in range(3)],
                100 * random.uniform(-1, 1)
                )
            rotations[j, :] = rot.get_quaternion()
            node.rotate(rot, origin=origin)

        mesh_2.rotate(rotations, origin=origin)

        string1 = mesh_1.get_string(header=False)
        string2 = mesh_2.get_string(header=False)
        self.compare_strings('Rotate_mesh_individual', string1, string2)

    def test_comments_in_solid(self):
        """
        Check if comments in the solid file are handled correctly if they are
        inside a mesh section.
        """

        ref_file = os.path.join(testing_input, 'comments_in_input_file_ref.dat')
        solid_file = os.path.join(testing_input, 'comments_in_input_file.dat')
        mesh = InputFile(dat_file=solid_file)

        # Add one element with BCs.
        mat = BaseMeshItem('material')
        sets = mesh.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
            [0, 0, 0], [1, 2, 3])
        mesh.add(BoundaryCondition(sets['start'], 'test',
            bc_type=mpy.dirichlet))
        mesh.add(BoundaryCondition(sets['end'], 'test', bc_type=mpy.neumann))

        string2 = mesh.get_string(header=False).strip()

        with open(ref_file, 'r') as r_file:
            string1 = r_file.read().strip()
        self.compare_strings('test_comments_in_solid', string1, string2)

    def test_curve_3d_helix(self):
        """Create a helix from a parametric curve."""

        # Ignore some strange warnings.
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and functions.
        mat = BaseMeshItem('material')

        # Set parameters for the helix.
        R = 2.
        tz = 1.  # incline
        n = 2  # number of turns
        n_el = 8

        # Create a helix with a parametric curve.
        def helix(t):
            return npAD.array([
                R * npAD.cos(t),
                R * npAD.sin(t),
                t * tz / (2 * np.pi)
                ])
        helix_set = input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat,
            helix, [0., 2. * np.pi * n], n_el=n_el)

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(helix_set['start'], 'BC1',
            bc_type=mpy.dirichlet))
        input_file.add(BoundaryCondition(helix_set['end'], 'BC2',
            bc_type=mpy.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input, 'curve_3d_helix_ref.dat')
        string2 = input_file.get_string(header=False).strip()
        with open(ref_file, 'r') as r_file:
            string1 = r_file.read().strip()
        self.compare_strings('test_curve_3d_helix', string1, string2)

    def test_curve_2d_sin(self):
        """Create a sin from a parametric curve."""

        # Ignore some strange warnings.
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and functions.
        mat = BaseMeshItem('material')

        # Set parameters for the sin.
        n_el = 8

        # Create a helix with a parametric curve.
        def sin(t):
            return npAD.array([t, npAD.sin(t)])
        sin_set = input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat,
            sin, [0., 2. * np.pi], n_el=n_el)

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(sin_set['start'], 'BC1',
            bc_type=mpy.dirichlet))
        input_file.add(BoundaryCondition(sin_set['end'], 'BC2',
            bc_type=mpy.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input, 'curve_2d_sin_ref.dat')
        string2 = input_file.get_string(header=False).strip()
        with open(ref_file, 'r') as r_file:
            string1 = r_file.read().strip()
        self.compare_strings('test_curve_2d_sin', string1, string2)

    def test_curve_3d_curve_rotation(self):
        """Create a line from a parametric curve and prescribe the rotation."""

        # AD.
        from autograd import jacobian

        # Ignore some strange warnings.
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and functions.
        mat = BaseMeshItem('material')

        # Set parameters for the line.
        L = 1.1
        n_el = 4

        # Create a helix with a parametric curve.
        def curve(t):
            return npAD.array([L * t, t * t * L * L, 0.])

        def rotation(t):
            rp2 = jacobian(curve)(t)
            rp = [rp2[0], rp2[1], 0]
            R1 = Rotation([1, 0, 0], t * 2 * np.pi)
            R2 = Rotation.from_basis(rp, [0, 0, 1])
            return R2 * R1
        sin_set = input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat,
            curve, [0., 1.], n_el=n_el, function_rotation=rotation)

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(sin_set['start'], 'BC1',
            bc_type=mpy.dirichlet))
        input_file.add(BoundaryCondition(sin_set['end'], 'BC2',
            bc_type=mpy.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input, 'curve_3d_line_rotation_ref.dat')
        string2 = input_file.get_string(header=False).strip()
        with open(ref_file, 'r') as r_file:
            string1 = r_file.read().strip()
        self.compare_strings('test_curve_3d_line_rotation', string1, string2)

    def test_vtk_writer(self):
        """Test the output created by the VTK writer."""

        # Initialize writer.
        writer = VTKWriter()

        # Add poly line.
        writer.add_poly_line([
            [0, 0, -2],
            [1, 1, -2],
            [2, 2, -1]
            ])

        # Add quadratic quad.
        cell_data = {}
        cell_data['cell_data_1'] = 3
        cell_data['cell_data_2'] = [66, 0, 1]
        point_data = {}
        point_data['point_data_1'] = [1, 2, 3, 4, 5, -2, -3, 0]
        point_data['point_data_2'] = [
                [0.25, 0, -0.25],
                [1, 0.25, 0],
                [2, 0, 0],
                [2.25, 1.25, 0.5],
                [2, 2.25, 0],
                [1, 2, 0.5],
                [0, 2.25, 0],
                [0, 1, 0.5]
            ]
        writer._add_cell(vtk.vtkQuadraticQuad,
            [
                [0.25, 0, -0.25],
                [1, 0.25, 0],
                [2, 0, 0],
                [2.25, 1.25, 0.5],
                [2, 2.25, 0],
                [1, 2, 0.5],
                [0, 2.25, 0],
                [0, 1, 0.5]
            ], [0, 2, 4, 6, 1, 3, 5, 7], cell_data=cell_data, point_data=point_data)

        # Add tetraeder.
        cell_data = {}
        cell_data['cell_data_2'] = [5, 0, 10]
        point_data = {}
        point_data['point_data_1'] = [1, 2, 3, 4]
        writer._add_cell(vtk.vtkTetra,
            [
                [3, 3, 3],
                [4, 4, 3],
                [4, 3, 3],
                [4, 4, 4]
            ], [0, 2, 1, 3], cell_data=cell_data, point_data=point_data)

        # Write to file.
        ref_file = os.path.join(testing_input, 'vtk_writer_test_ref.vtu')
        vtk_file = os.path.join(testing_temp, 'vtk_writer_test.vtu')
        writer.write_vtk(vtk_file, ascii=True)

        # Compare.
        if compare_xml(ref_file, vtk_file):
            self.assertTrue(True, '')
        else:
            # If the trivial compare CML function fails, compare the full
            # strings to see the differences.
            with open(ref_file, 'r') as r_file:
                string1 = r_file.read().strip()
            with open(vtk_file, 'r') as r_file:
                string2 = r_file.read().strip()
            self.compare_strings('test_vtk_writer', string1, string2)

    def test_vtk_writer_beam(self):
        """Create a sample mesh and check the VTK output."""

        # Create the mesh.
        mesh = Mesh()

        # Add content to the mesh.
        mat = Material('Material', 1., 1., 1., 0.1)
        mesh.create_beam_mesh_honeycomb(Beam3rHerm2Lin3, mat, 2., 5, 3, 2)

        # Write VTK output."""
        ref_file = os.path.join(testing_input, 'vtk_writer_beam_test_ref.vtu')
        vtk_file = os.path.join(testing_temp, 'vtk_writer_beam_test.vtu')
        mesh.write_vtk(os.path.join(testing_temp, vtk_file), ascii=True)

        # Compare.
        if compare_xml(ref_file, vtk_file):
            self.assertTrue(True, '')
        else:
            # If the trivial compare CML function fails, compare the full
            # strings to see the differences.
            with open(ref_file, 'r') as r_file:
                string1 = r_file.read().strip()
            with open(vtk_file, 'r') as r_file:
                string2 = r_file.read().strip()
            self.compare_strings('test_vtk_writer_beam', string1, string2)


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
            msg='Test {} failed!'.format(name))

        # If successful delete created files directory.
        if int(child.returncode) == 0:
            os.remove(input_file)
            items = glob.glob(testing_temp + '/xxx_' + name + '*')
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
            2.07e2,  # E-Modul
            0,  # nu
            1e-3,  # rho
            0.2,  # diameter of beam
            shear_correction=1.1
            )

        # Create the honeycomb mesh.
        mesh_honeycomb = Mesh()
        honeycomb_set = mesh_honeycomb.create_beam_mesh_honeycomb(
            Beam3rHerm2Lin3, material, 50.0, 10, 4, 1, closed_top=False)
        mesh_honeycomb.rotate(Rotation([0, 0, 1], 0.5 * np.pi))

        # Functions for the boundary conditions
        ft = Function(
            'COMPONENT 0 FUNCTION a\n'
            + 'VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 '
            + 'TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0'
            )
        mesh_honeycomb.add(ft)

        # Change the sets to lines, only for purpose of matching the test file
        honeycomb_set['bottom'].geo_type = mpy.line
        honeycomb_set['top'].geo_type = mpy.line
        mesh_honeycomb.add(
            BoundaryCondition(honeycomb_set['bottom'],
                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 '
                + 'FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.dirichlet))
        mesh_honeycomb.add(
            BoundaryCondition(honeycomb_set['top'],
                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 '
                + 'FUNCT 0 0 {} 0 0 0 0 0 0',
                format_replacement=[ft],
                bc_type=mpy.dirichlet))

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
            material, [2, 0, -5], [2, 0, 5], 3)

        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(
                cantilever_set['start'],  # bc set
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 ' + \
                'FUNCT 0 0 0 0 0 0 0 0 0',  # bc string
                bc_type=mpy.dirichlet
                )
            )
        input_file.add(
            BoundaryCondition(
                cantilever_set['end'],  # bc set
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 ' + \
                'FUNCT {} {} 0 0 0 0 0 0 0',  # bc string
                format_replacement=[cos, sin],
                bc_type=mpy.dirichlet
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
            MAXTIME                               10.0
            TOLRES                                1.0E-4
            TOLDISP                               1.0E-11
            NORM_RESF                             Abs
            NORM_DISP                             Abs
            NORMCOMBI_RESFDISP                    And
            MAXITER                               20
            ''')
        input_file.add(InputSection('STRUCTURAL DYNAMIC', 'NUMSTEP 1',
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
                mesh.translate(17 * np.array([1, 0, 0]))
                honeycomb_set = mesh.create_beam_mesh_honeycomb(
                    Beam3rHerm2Lin3, material, 10, 6, 3, n_el=2,
                    vertical=vertical, closed_top=closed_top)
                mesh.add(
                    BoundaryCondition(honeycomb_set['bottom'],
                        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL '
                        + '0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                        bc_type=mpy.dirichlet))
                mesh.add(
                    BoundaryCondition(honeycomb_set['top'],
                        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL '
                        + '{1} {1} {1} 0 0 0 0 0 0 FUNCT {0} {0} {0} 0 0 0 0 0 0',
                        format_replacement=[ft[counter], 0.0001],
                        bc_type=mpy.neumann,
                        double_nodes=mpy.double_nodes_remove))
                counter += 1

        # Add to input file and set testing results.
        input_file.add(mesh)
        input_file.add(InputSection(
            'RESULT DESCRIPTION',
            '''
            STRUCTURE DIS structure NODE 190 QUANTITY dispx VALUE 1.31917493321867280e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 190 QUANTITY dispy VALUE 1.99335191674617163e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 190 QUANTITY dispz VALUE 6.92208840492946759e-02 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 470 QUANTITY dispx VALUE 1.32983000589799755e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 470 QUANTITY dispy VALUE 2.00555448758363231e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 470 QUANTITY dispz VALUE 6.97002973386536134e-02 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 711 QUANTITY dispx VALUE 7.69274537804140734e-02 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 711 QUANTITY dispy VALUE 1.24993807967913248e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 711 QUANTITY dispz VALUE 5.86799642123171789e-02 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1071 QUANTITY dispx VALUE 6.98802967009408832e-02 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1071 QUANTITY dispy VALUE 1.09892595703934184e-01 TOLERANCE 1e-10
            STRUCTURE DIS structure NODE 1071 QUANTITY dispz VALUE 4.83525916448545312e-02 TOLERANCE 1e-10
            '''
            ))

        # Run the input file in Baci.
        self.run_baci_test('honeycomb-variants', input_file)


if __name__ == '__main__':
    unittest.main()
    pass
