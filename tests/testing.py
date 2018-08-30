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
    InputSection, MaterialReissner, MaterialBeam, Function, Beam3rHerm2Lin3, \
    BoundaryCondition, Node, BaseMeshItem, VTKWriter, compare_xml, Mesh, \
    find_close_nodes, find_close_nodes_binning


def get_default_paths(name, throw_error=True):
    """Look for and return a path to cubit or pre_exodus."""

    if name == 'baci-release':
        default_paths = [
            ['/home/ivo/baci/work/release/baci-release', os.path.isfile],
            ['/hdd/gitlab-runner/lib/baci-release/baci-release', os.path.isfile]
            ]
    else:
        raise ValueError('Type {} not implemented!'.format(name))

    # Check which path exists.
    for [path, function] in default_paths:
        if function(path):
            return path
    else:
        if throw_error:
            raise ValueError('Path for {} not found!'.format(name))
        else:
            return None


# Define the testing paths.
testing_path = os.path.abspath(os.path.dirname(__file__))
testing_input = os.path.join(testing_path, 'reference-files')
testing_temp = os.path.join(testing_path, 'testing-tmp')
baci_release = get_default_paths('baci-release')


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

        # Set default values for global parameters.
        mpy.set_default_values()

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

        # Set default values for global parameters.
        mpy.set_default_values()

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

        # Set default values for global parameters.
        mpy.set_default_values()

        vector = 10 * np.array([-1.234243, -2.334343, -1.123123])
        phi = -12.152101868665
        rot = Rotation(vector, phi)
        for i in range(2):
            self.assertTrue(rot == Rotation(vector, phi + 2 * i * np.pi))

        q = np.array([0.97862427, -0.0884585, -0.16730294, -0.0804945])
        self.assertTrue(Rotation(q) == Rotation(-q))

    def test_inverse_rotation(self):
        """Test the inv() function for rotations."""

        # Set default values for global parameters.
        mpy.set_default_values()

        # Define test rotation.
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

        # Set default values for global parameters.
        mpy.set_default_values()

        rot1 = Rotation([1, 2, 3], 2)
        rot2 = Rotation([0.1, -0.2, 2], np.pi / 5)

        rot21 = get_relative_rotation(rot1, rot2)

        self.assertTrue(rot2 == rot21 * rot1)


def create_test_mesh(mesh):
    """Fill the mesh with a couple of test nodes and elements."""

    # Set the seed for the pseudo random numbers
    random.seed(0)

    # Add material to mesh.
    material = MaterialReissner()
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

    def compare_strings(self, name, reference, compare):
        """
        Compare two stings. If they are not identical open kompare and show
        differences.
        """

        # Check if the input data is a file that exists.
        reference_is_file = os.path.isfile(reference)
        compare_is_file = os.path.isfile(compare)

        # Get the correct data
        if reference_is_file:
            with open(reference, 'r') as myfile:
                reference_string = myfile.read()
        else:
            reference_string = reference

        if compare_is_file:
            with open(compare, 'r') as myfile:
                compare_string = myfile.read()
        else:
            compare_string = compare

        # Check if the strings are equal, if not compare the differences and
        # fail the test.
        is_equal = reference_string.strip() == compare_string.strip()
        if not is_equal:

            # Check if temp directory exists, and creates it if necessary.
            os.makedirs(testing_temp, exist_ok=True)

            # Get the paths of the files to compare. If a string was given
            # create a file with the string in it.
            if reference_is_file:
                reference_file = reference
            else:
                reference_file = os.path.join(testing_temp,
                    '{}_reference.dat'.format(name))
                with open(reference_file, 'w') as input_file:
                    input_file.write(reference_string)

            if compare_is_file:
                compare_file = compare
            else:
                compare_file = os.path.join(testing_temp,
                    '{}_compare.dat'.format(name))
                with open(compare_file, 'w') as input_file:
                    input_file.write(compare_string)

            child = subprocess.Popen(
                ['kompare', reference_file, compare_file],
                stderr=subprocess.PIPE)
            child.communicate()

        # Pass or fail the test.
        self.assertTrue(is_equal, name)

    def test_mesh_rotations(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

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

        # Compare the output for the two meshes.
        self.compare_strings(
            'test_meshpy_rotate_mesh',
            mesh_1.get_string(header=False),
            mesh_2.get_string(header=False))

    def test_mesh_rotations_individual(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self, when an array is passed with different rotations.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

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

        # Compare the output for the two meshes.
        self.compare_strings(
            'test_meshpy_rotate_mesh_individually',
            mesh_1.get_string(header=False),
            mesh_2.get_string(header=False))

    def test_comments_in_solid(self):
        """Test case with full classical import."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_comments_in_input_file_reference.dat')
        self.create_comments_in_solid(ref_file, False)

    def test_comments_in_solid_full(self):
        """Test case with full solid import."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_comments_in_input_file_full_reference.dat')
        self.create_comments_in_solid(ref_file, True)

    def create_comments_in_solid(self, ref_file, full_import):
        """
        Check if comments in the solid file are handled correctly if they are
        inside a mesh section.
        """

        # Set default values for global parameters.
        mpy.set_default_values()
        mpy.import_mesh_full = full_import

        solid_file = os.path.join(testing_input,
            'test_meshpy_comments_in_input_file.dat')
        mesh = InputFile(dat_file=solid_file)

        # Add one element with BCs.
        mat = BaseMeshItem('material')
        sets = mesh.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
            [0, 0, 0], [1, 2, 3])
        mesh.add(BoundaryCondition(sets['start'], 'test',
            bc_type=mpy.dirichlet))
        mesh.add(BoundaryCondition(sets['end'], 'test', bc_type=mpy.neumann))

        # Compare the output of the mesh.
        if full_import:
            full_name = 'create_comments_in_solid_full'
        else:
            full_name = 'create_comments_in_solid'
        self.compare_strings(
            full_name,
            ref_file,
            mesh.get_string(header=False).strip())

    def test_find_close_nodes_binning(self):
        """
        Test if the find_close nodes_binnign and find_close nodes functions
        return the same results.
        """

        # Set the seed for the pseudo random numbers.
        random.seed(0)

        # Add random nodes to a cube with width 2. Randomly add nodes close to
        # existing nodes.
        eps_medium = 1e-5
        n_nodes = 1000
        coords = np.zeros([n_nodes, 3])
        for i in range(n_nodes):
            # Check if this one should be close to another one.
            if random.randint(0, 4) == 0 and i > 0:
                close_node = random.randint(0, i - 1)
                for j in range(3):
                    coords[i, j] = (coords[close_node, j]
                        + eps_medium * random.uniform(-1, 1))
            else:
                for j in range(3):
                    coords[i, j] = random.uniform(-1, 1)

        # Add nodes between the bins.
        coords[-1, :] = [0., 0., 0.]
        coords[-2, :] = [0., 0., 0.]
        coords[-3, :] = [0., 0., 0.5]
        coords[-4, :] = [0., 0., 0.5]
        coords[-5, :] = [0., 0.5, 0.]
        coords[-6, :] = [0., 0.5, 0.]
        coords[-7, :] = [0.5, 0., 0.]
        coords[-8, :] = [0.5, 0., 0.]
        coords[-9, :] = [0. + eps_medium, 0., 0.]
        coords[-10, :] = [0., 0. + eps_medium, 0.]
        coords[-11, :] = [0., 0., 0. + eps_medium]
        coords[-12, :] = [0. - eps_medium, 0., 0.]
        coords[-13, :] = [0., 0. - eps_medium, 0.]
        coords[-14, :] = [0., 0., 0. - eps_medium]

        has_partner, partner = find_close_nodes_binning(coords, 4, 4, 4,
            100 * eps_medium)
        has_partner_brute, partner = find_close_nodes(coords, 100 * eps_medium)
        self.assertTrue(np.array_equal(has_partner, has_partner_brute))

        has_partner, partner = find_close_nodes_binning(coords, 4, 4, 4,
            eps_medium)
        has_partner_brute, partner = find_close_nodes(coords, eps_medium)
        self.assertTrue(np.array_equal(has_partner, has_partner_brute))

        has_partner, partner = find_close_nodes_binning(coords, 4, 4, 4,
            0.01 * eps_medium)
        has_partner_brute, partner = find_close_nodes(coords,
            0.01 * eps_medium)
        self.assertTrue(np.array_equal(has_partner, has_partner_brute))

    def test_curve_3d_helix(self):
        """
        Create a helix from a parametric curve where the parameter is
        transformed so the arc length along the beam is not proportional to
        the parameter.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Ignore some strange warnings.
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and functions.
        mat = BaseMeshItem('material')

        # Set parameters for the helix.
        R = 2.
        tz = 4.  # incline
        n = 1  # number of turns
        n_el = 5

        # Create a helix with a parametric curve.
        def helix(t):
            factor = 2
            t_trans = (npAD.exp(factor * t / (2. * np.pi * n)) * t
                / npAD.exp(factor))
            return npAD.array([
                R * npAD.cos(t_trans),
                R * npAD.sin(t_trans),
                t_trans * tz / (2 * np.pi)
                ])
        helix_set = input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat,
            helix, [0., 2. * np.pi * n], n_el=n_el)

        # Compare the coordinates with the ones from mathematica.
        coordinates_mathematica = np.loadtxt(os.path.join(testing_input,
                'test_meshpy_curve_3d_helix_mathematica.csv'), delimiter=',')
        self.assertLess(
            np.linalg.norm(
                coordinates_mathematica - input_file.get_global_coordinates()),
            mpy.eps_pos,
            'test_meshpy_curve_3d_helix'
            )

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(helix_set['start'], 'BC1',
            bc_type=mpy.dirichlet))
        input_file.add(BoundaryCondition(helix_set['end'], 'BC2',
            bc_type=mpy.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_3d_helix_reference.dat')
        self.compare_strings(
            'test_meshpy_curve_3d_helix',
            ref_file,
            input_file.get_string(header=False))

    def test_curve_2d_sin(self):
        """Create a sin from a parametric curve."""

        # Set default values for global parameters.
        mpy.set_default_values()

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

        # Compare the coordinates with the ones from mathematica.
        coordinates_mathematica = np.loadtxt(os.path.join(testing_input,
                'test_meshpy_curve_2d_sin_mathematica.csv'), delimiter=',')
        self.assertLess(
            np.linalg.norm(
                coordinates_mathematica - input_file.get_global_coordinates()),
            mpy.eps_pos,
            'test_meshpy_curve_2d_sin'
            )

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(sin_set['start'], 'BC1',
            bc_type=mpy.dirichlet))
        input_file.add(BoundaryCondition(sin_set['end'], 'BC2',
            bc_type=mpy.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_2d_sin_reference.dat')
        self.compare_strings(
            'test_meshpy_curve_2d_sin',
            ref_file,
            input_file.get_string(header=False))

    def test_curve_3d_curve_rotation(self):
        """Create a line from a parametric curve and prescribe the rotation."""

        # Set default values for global parameters.
        mpy.set_default_values()

        # AD.
        from autograd import jacobian

        # Ignore some strange warnings.
        warnings.filterwarnings('ignore', message='numpy.dtype size changed')

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
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_3d_line_rotation_reference.dat')
        self.compare_strings(
            'test_meshpy_curve_3d_line_rotation',
            ref_file,
            input_file.get_string(header=False))

    def test_vtk_writer(self):
        """Test the output created by the VTK writer."""

        # Set default values for global parameters.
        mpy.set_default_values()

        # Initialize writer.
        writer = VTKWriter()

        # Add poly line.
        writer.add_cell(vtk.vtkPolyLine, [
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
        writer.add_cell(vtk.vtkQuadraticQuad,
            [
                [0.25, 0, -0.25],
                [1, 0.25, 0],
                [2, 0, 0],
                [2.25, 1.25, 0.5],
                [2, 2.25, 0],
                [1, 2, 0.5],
                [0, 2.25, 0],
                [0, 1, 0.5]
            ], [0, 2, 4, 6, 1, 3, 5, 7],
            cell_data=cell_data, point_data=point_data)

        # Add tetraeder.
        cell_data = {}
        cell_data['cell_data_2'] = [5, 0, 10]
        point_data = {}
        point_data['point_data_1'] = [1, 2, 3, 4]
        writer.add_cell(vtk.vtkTetra,
            [
                [3, 3, 3],
                [4, 4, 3],
                [4, 3, 3],
                [4, 4, 4]
            ], [0, 2, 1, 3], cell_data=cell_data, point_data=point_data)

        # Write to file.
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_writer_reference.vtu')
        vtk_file = os.path.join(testing_temp,
            'test_meshpy_vtk_writer.vtu')
        writer.write_vtk(vtk_file, ascii=True)

        # Compare.
        if compare_xml(ref_file, vtk_file):
            self.assertTrue(True, '')
        else:
            # If the trivial compare CML function fails, compare the full
            # strings to see the differences.
            self.compare_strings('test_meshpy_vtk_writer', ref_file,
                vtk_file)

    def test_vtk_writer_beam(self):
        """Create a sample mesh and check the VTK output."""

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create the mesh.
        mesh = Mesh()

        # Add content to the mesh.
        mat = MaterialBeam(radius=0.05)
        mesh.create_beam_mesh_honeycomb(Beam3rHerm2Lin3, mat, 2., 2, 3, 2,
            add_sets=True)

        # Write VTK output."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_beam_reference.vtu')
        vtk_file = os.path.join(testing_temp, 'test_meshpy_vtk_beam.vtu')
        mesh.write_vtk(output_name='test_meshpy_vtk',
            output_directory=testing_temp, ascii=True)

        # Compare.
        if compare_xml(ref_file, vtk_file):
            self.assertTrue(True, '')
        else:
            # If the trivial compare CML function fails, compare the full
            # strings to see the differences.
            self.compare_strings('test_vtk_writer_beam', ref_file, vtk_file)

    def test_vtk_writer_solid(self):
        """Import a solid mesh and check the VTK output."""

        # Set default values for global parameters.
        mpy.set_default_values()

        # Without this parameter no solid VTK file would be written.
        mpy.import_mesh_full = True

        # Create the input file and read solid mesh data.
        input_file = InputFile()
        input_file.read_dat(os.path.join(testing_input, 'baci_input_tube.dat'))

        # Write VTK output."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_solid_reference.vtu')
        vtk_file = os.path.join(testing_temp, 'test_meshpy_vtk_solid.vtu')
        if os.path.isfile(vtk_file):
            os.remove(vtk_file)
        input_file.write_vtk(output_name='test_meshpy_vtk',
            output_directory=testing_temp, ascii=True)

        # Compare.
        if compare_xml(ref_file, vtk_file):
            self.assertTrue(True, '')
        else:
            # If the trivial compare CML function fails, compare the full
            # strings to see the differences.
            self.compare_strings('test_meshpy_vtk_solid', ref_file, vtk_file)


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
        os.makedirs(testing_temp, exist_ok=True)

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

    def test_honeycomb_sphere_as_input(self):
        """
        Test the honeycomb sphere model with different types of mesh import.
        """

        mpy.set_default_values()
        self.create_honeycomb_sphere_as_input('honeycomb_sphere')

        mpy.set_default_values()
        mpy.import_mesh_full = not mpy.import_mesh_full
        self.create_honeycomb_sphere_as_input('honeycomb_sphere_full_input')

    def create_honeycomb_sphere_as_input(self, name):
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
            dat_file=os.path.join(testing_input,
                'baci_input_honeycomb_sphere.dat'))

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
        material = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Create the honeycomb mesh.
        mesh_honeycomb = Mesh()
        honeycomb_set = mesh_honeycomb.create_beam_mesh_honeycomb(
            Beam3rHerm2Lin3, material, 50.0, 10, 4, 1, closed_top=False,
            add_sets=True)
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
        self.run_baci_test(name, input_file)

    def test_beam_and_solid_tube(self):
        """
        Test the honeycomb sphere model with different types of mesh import.
        """

        mpy.set_default_values()
        self.create_beam_and_solid_tube('beam_and_solid_tube')

        mpy.set_default_values()
        mpy.import_mesh_full = not mpy.import_mesh_full
        self.create_beam_and_solid_tube('beam_and_solid_tube')

    def create_beam_and_solid_tube(self, name):
        """Merge a solid tube with a beam tube and simulate them together."""

        # Create the input file and read solid mesh data.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Solid tube with beam tube')
        input_file.read_dat(os.path.join(testing_input, 'baci_input_tube.dat'))

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
        material = MaterialReissner(
            youngs_modulus=1e9,
            radius=0.25,
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

        # Call get_unique_geometry_sets to check that this does not affect the
        # mesh creation.
        input_file.get_unique_geometry_sets(link_nodes=True)

        # Run the input file in Baci.
        self.run_baci_test(name, input_file)

    def test_honeycomb_variants(self):
        """
        Create a few different honeycomb structures.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

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
        material = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
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
                        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL {1} {1} {1} '
                        + '0 0 0 0 0 0 FUNCT {0} {0} {0} 0 0 0 0 0 0',
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
        self.run_baci_test('honeycomb_variants', input_file)


if __name__ == '__main__':

    # Delete all files in the testing directory, if it exists.
    if os.path.isdir(testing_temp):
        for the_file in os.listdir(testing_temp):
            file_path = os.path.join(testing_temp, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    # Perform tests.
    unittest.main()
