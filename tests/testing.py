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
    find_close_nodes, find_close_nodes_binning, GeometryName, GeometrySet

# Geometry functions.
from meshpy.mesh_creation_functions.beam_basic_geometry import \
    create_beam_mesh_line, create_beam_mesh_arc_segment
from meshpy.mesh_creation_functions.beam_honeycomb import \
    create_beam_mesh_honeycomb, create_beam_mesh_honeycomb_flat
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve

# Global variable if this test is run by GitLab.
if ('TESTING_GITLAB' in os.environ.keys()
        and os.environ['TESTING_GITLAB'] == '1'):
    TESTING_GITLAB = True
else:
    TESTING_GITLAB = False


def get_default_paths(name):
    """Look for and return a path to baci-release."""

    if name == 'baci-release':
        default_paths = [
            ['/home/ivo/workspace/baci/release/baci-release', os.path.isfile],
            #['/home/ivo/baci/work/release/baci-release', os.path.isfile],
            ['/hdd/gitlab-runner/lib/baci-release/baci-release',
                os.path.isfile]
            ]
    else:
        raise ValueError('Type {} not implemented!'.format(name))

    # Check which path exists.
    for [path, function] in default_paths:
        if function(path):
            return path
    else:
        # In the case that no path was found, check if the script is performed
        # by a GitLab runner.
        if TESTING_GITLAB:
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

        rot = Rotation.from_rotation_vector(vector)
        q = rot.q
        self.assertTrue(rot == Rotation(-q))
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
        (rot * rot.inv()).get_rotation_vector()

    def test_relative_roation(self):
        """Test the relative rotation between two rotations."""

        # Set default values for global parameters.
        mpy.set_default_values()

        rot1 = Rotation([1, 2, 3], 2)
        rot2 = Rotation([0.1, -0.2, 2], np.pi / 5)

        rot21 = get_relative_rotation(rot1, rot2)

        self.assertTrue(rot2 == rot21 * rot1)

    def test_rotation_vector(self):
        """Test if the rotation vector functions give a correct result."""

        # Calculate rotation vector and quaternion.
        axis = np.array([1.36568, -2.96784, 3.23346878])
        angle = 0.7189467
        rotation_vector = angle * axis / np.linalg.norm(axis)
        q = np.zeros(4)
        q[0] = np.cos(angle / 2)
        q[1:] = np.sin(angle / 2) * axis / np.linalg.norm(axis)

        # Check that the rotation object from the quaternion and rotation
        # vector are equal.
        rotation_from_vec = Rotation.from_rotation_vector(rotation_vector)
        self.assertTrue(Rotation(q) == rotation_from_vec)
        self.assertTrue(Rotation(axis, angle) == rotation_from_vec)

        # Check that the same rotation vector is returned after being converted
        # to a quaternion.
        self.assertLess(
            np.linalg.norm(
                rotation_vector - rotation_from_vec.get_rotation_vector()),
            mpy.eps_quaternion,
            'test_rotation_vector'
            )

    def test_rotation_operator_overload(self):
        """Test if the operator overloading gives a correct result."""

        # Calculate rotation and vector.
        axis = np.array([1.36568, -2.96784, 3.23346878])
        angle = 0.7189467
        rot = Rotation(axis, angle)
        vector = [2.234234, -4.213234, 6.345234]

        # Check the result of the operator overloading.
        result_vector = np.dot(rot.get_rotation_matrix(), vector)
        self.assertLess(
            np.linalg.norm(
                result_vector - rot * vector),
            mpy.eps_quaternion,
            'test_rotation_vector'
            )
        self.assertLess(
            np.linalg.norm(
                result_vector - rot * np.array(vector)),
            mpy.eps_quaternion,
            'test_rotation_vector'
            )

        # Check multiplication with None.
        self.assertTrue(rot == rot * None)
        self.assertTrue(rot == rot.copy() * None)

    def test_rotation_matrix(self):
        """
        Test if the correct quaternions are generated from a rotation matrix.
        """

        # Do one calculation for each case in
        # Rotation().from_rotation_matrix().
        vectors = [
            [[1, 0, 0], [0, -1, 0]],
            [[0, 0, 1], [0, 1, 0]],
            [[-1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1]]
            ]

        for t1, t2 in vectors:
            rot = Rotation().from_basis(t1, t2)
            t1_rot = rot * [1, 0, 0]
            t2_rot = rot * [0, 1, 0]
            self.assertLess(np.linalg.norm(t1 - t1_rot), mpy.eps_quaternion,
                'test_rotation_matrix: compare t1')
            self.assertLess(np.linalg.norm(t2 - t2_rot), mpy.eps_quaternion,
                'test_rotation_matrix: compare t2')


def create_test_mesh(mesh):
    """Fill the mesh with a couple of test nodes and elements."""

    # Set the seed for the pseudo random numbers
    random.seed(0)

    # Add material to mesh.
    material = MaterialReissner()
    mesh.add(material)

    # Add three test nodes and add them to a beam element
    for _j in range(3):
        mesh.add(Node(
            [100 * random.uniform(-1, 1) for _i in range(3)],
            rotation=Rotation(
                [100 * random.uniform(-1, 1) for _i in range(3)],
                100 * random.uniform(-1, 1)
                )))
    beam = Beam3rHerm2Lin3(material=material, nodes=mesh.nodes)
    mesh.add(beam)

    # Add a beam line with three elements
    create_beam_mesh_line(mesh, Beam3rHerm2Lin3, material,
        [100 * random.uniform(-1, 1) for _i in range(3)],
        [100 * random.uniform(-1, 1) for _i in range(3)],
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
        if not is_equal and not TESTING_GITLAB:

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
            [100 * random.uniform(-1, 1) for _i in range(3)],
            100 * random.uniform(-1, 1)
            )
        origin = [100 * random.uniform(-1, 1) for _i in range(3)]

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
        origin = [100 * random.uniform(-1, 1) for _i in range(3)]
        for j, node in enumerate(mesh_1.nodes):
            rot = Rotation(
                [100 * random.uniform(-1, 1) for _i in range(3)],
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

    def test_mesh_reflection(self):
        """Check the Mesh().reflect function."""

        def compare_reflection(origin=False, flip=False):
            """
            Create a mesh, and its mirrored counterpart and then compare the
            dat files.
            """

            # Rotations to be applied.
            rot_1 = Rotation([0, 1, 1], np.pi / 6)
            rot_2 = Rotation([1, 2.455, -1.2324], 1.2342352)

            mesh_ref = InputFile(maintainer='Ivo Steinbrecher')
            mesh = InputFile(maintainer='Ivo Steinbrecher')
            mat = MaterialReissner(radius=0.1)

            # Create the reference mesh.
            if not flip:
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
                    [0, 0, 0], [1, 0, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
                    [1, 0, 0], [1, 1, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
                    [1, 1, 0], [1, 1, 1], n_el=1)
            else:
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
                    [1, 0, 0], [0, 0, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
                    [1, 1, 0], [1, 0, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
                    [1, 1, 1], [1, 1, 0], n_el=1)

                # Reorder the internal nodes.
                old = mesh_ref.nodes.copy()
                mesh_ref.nodes[0] = old[2]
                mesh_ref.nodes[2] = old[0]
                mesh_ref.nodes[3] = old[5]
                mesh_ref.nodes[5] = old[3]
                mesh_ref.nodes[6] = old[8]
                mesh_ref.nodes[8] = old[6]

            mesh_ref.rotate(rot_1)

            # Create the mesh that will be mirrored.
            create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat, [0, 0, 0],
                [-1, 0, 0], n_el=1)
            create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat, [-1, 0, 0],
                [-1, 1, 0], n_el=1)
            create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat, [-1, 1, 0],
                [-1, 1, 1], n_el=1)
            mesh.rotate(rot_1.inv())

            # Rotate everything, to show generalised reflection.
            mesh_ref.rotate(rot_2)
            mesh.rotate(rot_2)

            if origin:
                # Translate everything so the reflection plane is not in the
                # origin.
                r = [1, 2.455, -1.2324]
                mesh_ref.translate(r)
                mesh.translate(r)
                mesh.reflect(2 * (rot_2 * [1, 0, 0]), origin=r, flip=flip)
            else:
                mesh.reflect(2 * (rot_2 * [1, 0, 0]), flip=flip)

            # Compare the dat files.
            self.compare_strings(
                'test_meshpy_reflect_origin_{}_flip_{}'.format(origin, flip),
                mesh_ref.get_string(header=False),
                mesh.get_string(header=False))

        # Compare all 4 possible variations.
        for flip in [True, False]:
            for origin in [True, False]:
                compare_reflection(origin=origin, flip=flip)

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
        sets = create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat,
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

        # Set default values for global parameters.
        mpy.set_default_values()

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

        has_partner, _partner = find_close_nodes_binning(coords, 4, 4, 4,
            100 * eps_medium)
        has_partner_brute, _partner = find_close_nodes(
            coords, 100 * eps_medium)
        self.assertTrue(np.array_equal(has_partner, has_partner_brute))

        has_partner, _partner = find_close_nodes_binning(coords, 4, 4, 4,
            eps_medium)
        has_partner_brute, _partner = find_close_nodes(coords, eps_medium)
        self.assertTrue(np.array_equal(has_partner, has_partner_brute))

        has_partner, _partner = find_close_nodes_binning(coords, 4, 4, 4,
            0.01 * eps_medium)
        has_partner_brute, _partner = find_close_nodes(coords,
            0.01 * eps_medium)
        self.assertTrue(np.array_equal(has_partner, has_partner_brute))

    def test_find_close_nodes_binning_flat(self):
        """
        Test case for coupling of nodes, when the nodes are all on a plane.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Dummy material.
        material = MaterialReissner(radius=0.1)

        def create_flat_mesh():
            """Create a flat honeycomb mesh."""
            input_file = InputFile()
            create_beam_mesh_honeycomb_flat(input_file, Beam3rHerm2Lin3,
                material, 1, 5, 5,
                create_couplings=False)
            return input_file

        # Get a reference solution for the coupling nodes.
        reference_partners, _temp = create_flat_mesh().get_close_nodes(
            nodes='all', binning=False, return_nodes=False)

        # Apply different rotations and compare the partner results.
        rotations = [
            Rotation([1, 0, 0], np.pi * 0.5),
            Rotation([0, 1, 0], np.pi * 0.5),
            Rotation([0, 0, 1], np.pi * 0.5),
            Rotation([1, 3, -4], 25.21561 * np.pi * 0.5)
            ]
        for rotation in rotations:
            # Create the input file.
            input_file = create_flat_mesh()
            input_file.rotate(rotation)
            partners, _temp = input_file.get_close_nodes(nodes='all',
                return_nodes=False)

            # Compare the partners with the reference.
            self.assertTrue(np.array_equal(
                np.array(reference_partners),
                np.array(partners)
                ))

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
        helix_set = create_beam_mesh_curve(input_file, Beam3rHerm2Lin3, mat,
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
        sin_set = create_beam_mesh_curve(input_file, Beam3rHerm2Lin3, mat,
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
        sin_set = create_beam_mesh_curve(input_file, Beam3rHerm2Lin3, mat,
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

    def test_curve_3d_line(self):
        """
        Create a line from a parametric curve. Once the interval is in
        ascending order, once in descending. This tests checks that the
        elements are created with the correct tangent vectors.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Set header
        input_file.set_default_header_static(time_step=0.05, n_steps=20)

        # Add material and function.
        mat = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)
        ft = Function('COMPONENT 0 FUNCTION t')
        input_file.add(ft)

        # Create a line with a parametric curve (and a transformed parameter).
        def line(t):
            factor = 2
            t_trans = (npAD.exp(factor * t / (2. * np.pi)) * t
                / npAD.exp(factor))
            return npAD.array([t_trans, 0, 0])

        # Create mesh.
        set_1 = create_beam_mesh_curve(input_file, Beam3rHerm2Lin3, mat, line,
            [0., 5.], n_el=3)
        input_file.translate([0, 1, 0])
        set_2 = create_beam_mesh_curve(input_file, Beam3rHerm2Lin3, mat, line,
            [5., 0.], n_el=3)

        # Add boundary conditions.
        bc_list = [
            [set_1, ['start', 'end']],
            [set_2, ['end', 'start']]
            ]
        force_fac = 0.01
        for set_item, [name_start, name_end] in bc_list:
            input_file.add(BoundaryCondition(set_item[name_start],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.dirichlet))
            input_file.add(BoundaryCondition(set_item[name_end],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '{1} {1} {1} {1} {1} {1} 0 0 0 FUNCT {0} {0} {0} {0} {0} {0}'
                + ' 0 0 0',
                bc_type=mpy.neumann,
                format_replacement=[ft, force_fac]))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_3d_line_reference.dat')
        self.compare_strings(
            'test_meshpy_curve_3d_line',
            ref_file,
            input_file.get_string(header=False))

    def test_segment(self):
        """Create a circular segment and compare it with the reference file."""

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and function.
        mat = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Create mesh.
        mesh = create_beam_mesh_arc_segment(input_file, Beam3rHerm2Lin3, mat,
            [3, 6, 9.2], Rotation([4.5, 7, 10], np.pi / 5), 10, np.pi / 2.3,
            n_el=5)

        # Add boundary conditions.
        input_file.add(BoundaryCondition(mesh['start'],
            'rb', bc_type=mpy.dirichlet))
        input_file.add(BoundaryCondition(mesh['end'],
            'rb', bc_type=mpy.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_segment_reference.dat')
        self.compare_strings(
            'test_meshpy_segment',
            ref_file,
            input_file.get_string(header=False))

    def test_close_beam(self):
        """
        Create a circle with different methods.
        - Create the mesh manually by creating the nodes and connecting them to
          the elements.
        - Create one full circle and connect it to its beginning.
        - Create two half circle and connect their start / end nodes.
        All of those methods should give the exact same mesh.
        Both variants are also tried with different rotations at the beginning.
        """

        # Ignore some strange warnings.
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")

        # Parameters for this test case.
        n_el = 3
        R = 1.235
        additional_rotation = Rotation([0, 1, 0], 0.5)

        # Set default values for global parameters.
        mpy.set_default_values()

        # Define material.
        mat = MaterialReissner(radius=0.1)

        def create_mesh_manually(start_rotation):
            """Create the full circle manually."""
            input_file = InputFile(maintainer='Ivo Steinbrecher')
            input_file.add(mat)

            # Add nodes.
            for i in range(4 * n_el):
                basis = start_rotation * Rotation([0, 0, 1], np.pi * 0.5)
                r = [R, 0, 0]
                node = Node(r, basis)
                rotation = Rotation([0, 0, 1], 0.5 * i * np.pi / n_el)
                node.rotate(rotation, [0, 0, 0])
                input_file.nodes.append(node)

            # Add elements.
            for i in range(2 * n_el):
                node_index = [2 * i, 2 * i + 1, 2 * i + 2]
                nodes = []
                for index in node_index:
                    if index == len(input_file.nodes):
                        nodes.append(input_file.nodes[0])
                    else:
                        nodes.append(input_file.nodes[index])
                element = Beam3rHerm2Lin3(mat, nodes)
                input_file.add(element)

            # Add sets.
            geom_set = GeometryName()
            geom_set['start'] = GeometrySet(mpy.point,
                nodes=input_file.nodes[0])
            geom_set['end'] = GeometrySet(mpy.point, nodes=input_file.nodes[0])
            geom_set['line'] = GeometrySet(mpy.line, nodes=input_file.nodes)
            input_file.add(geom_set)
            return input_file

        def one_full_circle_closed(function, argument_list,
                additional_rotation=None):
            """Create one full circle and connect it to itself."""

            input_file = InputFile(maintainer='Ivo Steinbrecher')

            if additional_rotation is not None:
                start_rotation = additional_rotation * \
                    Rotation([0, 0, 1], np.pi * 0.5)
                input_file.add(Node([R, 0, 0], rotation=start_rotation))
                function(input_file, start_node=input_file.nodes[0],
                    end_node=True, add_sets=True, **(argument_list))
            else:
                function(input_file, end_node=True, add_sets=True,
                    **(argument_list))
            return input_file

        def two_half_circles_closed(function, argument_list,
                additional_rotation=None):
            """
            Create two half circles and close them, by reusing the connecting
            nodes.
            """

            input_file = InputFile(maintainer='Ivo Steinbrecher')

            if additional_rotation is not None:
                start_rotation = additional_rotation * \
                    Rotation([0, 0, 1], np.pi * 0.5)
                input_file.add(Node([R, 0, 0], rotation=start_rotation))
                set_1 = function(input_file, start_node=input_file.nodes[0],
                    **(argument_list[0]))
            else:
                set_1 = function(input_file, **(argument_list[0]))

            set_2 = function(input_file, start_node=set_1['end'],
                end_node=set_1['start'], **(argument_list[1]))

            # Add sets.
            geom_set = GeometryName()
            geom_set['start'] = GeometrySet(mpy.point, nodes=set_1['start'])
            geom_set['end'] = GeometrySet(mpy.point, nodes=set_2['end'])
            geom_set['line'] = GeometrySet(mpy.line,
                nodes=[set_1['line'], set_2['line']], filter_double_nodes=True)
            input_file.add(geom_set)

            return input_file

        def get_arguments_arc_segment(circle_type):
            """Return the arguments for the arc segment function."""
            if circle_type == 0:
                # Full circle.
                arg_rot_angle = np.pi / 2
                arg_angle = 2 * np.pi
                arg_n_el = 2 * n_el
            elif circle_type == 1:
                # First half circle.
                arg_rot_angle = np.pi / 2
                arg_angle = np.pi
                arg_n_el = n_el
            elif circle_type == 2:
                # Second half circle.
                arg_rot_angle = 3 * np.pi / 2
                arg_angle = np.pi
                arg_n_el = n_el
            return {
                'beam_object': Beam3rHerm2Lin3,
                'material': mat,
                'center': [0, 0, 0],
                'axis_rotation': Rotation([0, 0, 1], arg_rot_angle),
                'radius': R,
                'angle': arg_angle,
                'n_el': arg_n_el
                }

        def circle_function(t):
            """Function for the circle."""
            return R * npAD.array([npAD.cos(t), npAD.sin(t)])

        def get_arguments_curve(circle_type):
            """Return the arguments for the curve function."""
            if circle_type == 0:
                # Full circle.
                arg_interval = [0, 2 * np.pi]
                arg_n_el = 2 * n_el
            elif circle_type == 1:
                # First half circle.
                arg_interval = [0, np.pi]
                arg_n_el = n_el
            elif circle_type == 2:
                # Second half circle.
                arg_interval = [np.pi, 2 * np.pi]
                arg_n_el = n_el
            return {
                'beam_object': Beam3rHerm2Lin3,
                'material': mat,
                'function': circle_function,
                'interval': arg_interval,
                'n_el': arg_n_el
                }

        # Check the meshes without additional rotation.
        ref_file = os.path.join(testing_input,
            'test_meshpy_close_beam_reference.dat')

        self.compare_strings(
            'test_meshpy_close_beam_manual',
            ref_file,
            create_mesh_manually(Rotation()).get_string(header=False))
        self.compare_strings(
            'test_meshpy_close_beam_full_segment',
            ref_file,
            one_full_circle_closed(create_beam_mesh_arc_segment,
                get_arguments_arc_segment(0)).get_string(header=False))
        self.compare_strings(
            'test_meshpy_close_beam_split_segment',
            ref_file,
            two_half_circles_closed(create_beam_mesh_arc_segment,
                [get_arguments_arc_segment(1), get_arguments_arc_segment(2)]
                ).get_string(header=False))
        self.compare_strings(
            'test_meshpy_close_beam_full_curve',
            ref_file,
            one_full_circle_closed(create_beam_mesh_curve,
                get_arguments_curve(0)).get_string(header=False))
        self.compare_strings(
            'test_meshpy_close_beam_split_curve',
            ref_file,
            two_half_circles_closed(create_beam_mesh_curve,
                [get_arguments_curve(1), get_arguments_curve(2)]
                ).get_string(header=False))

        # Check the meshes with additional rotation.
        ref_file = os.path.join(testing_input,
            'test_meshpy_close_beam_rotated_reference.dat')

        self.compare_strings(
            'test_meshpy_close_beam_manual_rotation',
            ref_file,
            create_mesh_manually(additional_rotation).get_string(header=False))
        self.compare_strings(
            'test_meshpy_close_beam_full_segment_rotation',
            ref_file,
            one_full_circle_closed(create_beam_mesh_arc_segment,
                get_arguments_arc_segment(0),
                additional_rotation=additional_rotation).get_string(
                    header=False))
        self.compare_strings(
            'test_meshpy_close_beam_split_segment_rotation',
            ref_file,
            two_half_circles_closed(create_beam_mesh_arc_segment,
                [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
                additional_rotation=additional_rotation
                ).get_string(header=False))
        self.compare_strings(
            'test_meshpy_close_beam_full_curve_rotation',
            ref_file,
            one_full_circle_closed(create_beam_mesh_curve,
                get_arguments_curve(0),
                additional_rotation=additional_rotation).get_string(
                    header=False))
        self.compare_strings(
            'test_meshpy_close_beam_split_curve_rotation',
            ref_file,
            two_half_circles_closed(create_beam_mesh_curve,
                [get_arguments_curve(1), get_arguments_curve(2)],
                additional_rotation=additional_rotation
                ).get_string(header=False))

    def test_replace_nodes(self):
        """Test case for coupling of nodes, and reusing the identical nodes."""

        mat = MaterialReissner(radius=0.1, youngs_modulus=1)
        rot = Rotation([1, 2, 43], 213123)

        def create_mesh():
            """Create two empty meshes."""
            name = 'Ivo Steinbrecher'
            return InputFile(maintainer=name), InputFile(maintainer=name)

        # Create a beam with two elements. Once immediately and once as two
        # beams with couplings.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [1, 0, 0],
            [2, 0, 0])

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the coupling mesh.
        mesh_couple.couple_nodes(coupling_type=mpy.coupling_fix_reuse)

        # Compare the meshes.
        self.compare_strings('test_replace_nodes_case_1',
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False))

        # Create two overlapping beams. This is to test that the middle nodes
        # are not coupled.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        set_ref = create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat,
            [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [1, 0, 0], start_node=set_ref['start'], end_node=set_ref['end'])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [1, 0, 0])

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the coupling mesh.
        mesh_couple.couple_nodes(coupling_type=mpy.coupling_fix_reuse)

        # Compare the meshes.
        self.compare_strings('test_replace_nodes_case_2',
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False))

        # Create a beam with two elements. Once immediately and once as two
        # beams with couplings.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [0, 0, 0],
            [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [1, 0, 0],
            [2, 0, 0])

        # Create set with all the beam nodes.
        node_set_1_ref = GeometrySet(mpy.line, nodes=mesh_ref.nodes)
        node_set_2_ref = GeometrySet(mpy.line, nodes=mesh_ref.nodes)
        node_set_1_couple = GeometrySet(mpy.line, nodes=mesh_couple.nodes)
        node_set_2_couple = GeometrySet(mpy.line, nodes=mesh_couple.nodes)

        # Create connecting beams.
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat, [1, 0, 0],
            [2, 2, 2])
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Lin3, mat, [1, 0, 0],
            [2, -2, -2])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [1, 0, 0],
            [2, 2, 2])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Lin3, mat, [1, 0, 0],
            [2, -2, -2])

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the mesh.
        mesh_ref.couple_nodes(coupling_type=mpy.coupling_fix)
        mesh_couple.couple_nodes(coupling_type=mpy.coupling_fix_reuse)

        # Add the node sets.
        mesh_ref.add(node_set_1_ref)
        mesh_couple.add(node_set_1_couple)

        # Add BCs.
        mesh_ref.add(BoundaryCondition(node_set_2_ref, 'BC1',
            bc_type=mpy.neumann))
        mesh_couple.add(BoundaryCondition(node_set_2_couple, 'BC1',
            bc_type=mpy.neumann))

        # Compare the meshes.
        self.compare_strings('test_replace_nodes_case_3',
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False))

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
        create_beam_mesh_honeycomb(mesh, Beam3rHerm2Lin3, mat, 2., 2, 3, n_el=2,
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

        # Write VTK output.
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

    def test_vtk_writer_solid_elements(self):
        """
        Import a solid mesh with all solid types and check the VTK output.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Without this parameter no solid VTK file would be written.
        mpy.import_mesh_full = True

        # Create the input file and read solid mesh data.
        input_file = InputFile()
        input_file.read_dat(
            os.path.join(testing_input, 'baci_input_solid_elements.dat'))

        # Write VTK output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_solid_elements_reference.vtu')
        vtk_file = os.path.join(
            testing_temp, 'test_meshpy_vtk_elements_solid.vtu')
        if os.path.isfile(vtk_file):
            os.remove(vtk_file)
        input_file.write_vtk(output_name='test_meshpy_vtk_elements',
            output_directory=testing_temp, ascii=True)

        # Compare.
        if compare_xml(ref_file, vtk_file):
            self.assertTrue(True, '')
        else:
            # If the trivial compare CML function fails, compare the full
            # strings to see the differences.
            self.compare_strings(
                'test_meshpy_vtk_elements_solid', ref_file, vtk_file)


class TestFullBaci(unittest.TestCase):
    """
    Create and run input files in Baci. They are test files and Baci should
    return 0.
    """

    def setUp(self):
        """
        This method is called before each test. Check if BACI was found.
        If not, skip the test.
        """
        if baci_release is None:
            self.skipTest('BACI path was not found!')

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

        # Delete the results given in the input file.
        input_file.delete_section('RESULT DESCRIPTION')
        input_file.add('-----RESULT DESCRIPTION')

        # Add result checks.
        displacement = [0.0, -8.09347205557697258, 2.89298034569662965]

        nodes = [268, 188, 182]
        for node in nodes:
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[i]
                        )
                    )
                )

        # Material for the beam.
        material = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Create the honeycomb mesh.
        mesh_honeycomb = Mesh()
        honeycomb_set = create_beam_mesh_honeycomb(mesh_honeycomb,
            Beam3rHerm2Lin3, material, 50.0, 10, 4, n_el=1, closed_top=False,
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
        cantilever_set = create_beam_mesh_line(input_file, Beam3rHerm2Lin3,
            material, [2, 0, -5], [2, 0, 5], n_el=3)

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

        # Add result checks.
        displacement = [
            [1.50796091342925, 1.31453288915877e-8, 0.0439008100184687],
            [0.921450108160878, 1.41113401669104e-15, 0.0178350143764099]
            ]

        nodes = [35, 69]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[j][i]
                        )
                    )
                )

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
                honeycomb_set = create_beam_mesh_honeycomb(mesh,
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

        # Add mesh to input file.
        input_file.add(mesh)

        # Add result checks.
        displacement = [
            [1.319174933218672e-1, 1.993351916746171e-1, 6.922088404929467e-2],
            [1.329830005897997e-1, 2.005554487583632e-1, 6.970029733865361e-2],
            [7.692745378041407e-2, 1.249938079679132e-1, 5.867996421231717e-2],
            [6.988029670094088e-2, 1.098925957039341e-1, 4.835259164485453e-2]
            ]

        nodes = [190, 470, 711, 1071]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[j][i]
                        )
                    )
                )

        # Run the input file in Baci.
        self.run_baci_test('honeycomb_variants', input_file)

    def test_rotated_beam_axis(self):
        """
        Create three beams that consist of two connected lines.
        - The first case uses the same nodes for the connection of the lines,
          and the nodes are equal in this case.
        - The second case uses the same nodes for the connection of the lines,
          but the nodes have a different rotation along the basis vector 1.
        - The third case uses two nodes at the connection between the lines,
          and couples them with a coupling.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Rotation of beam along axis'
            )

        # Set header
        input_file.set_default_header_static(
            time_step=0.05,
            n_steps=20,
            binning_bounding_box=[-10, -10, -10, 10, 10, 10]
            )

        # Define linear function over time.
        ft = Function('COMPONENT 0 FUNCTION t')
        input_file.add(ft)

        # Set beam material.
        mat = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Direction of the lines and the rotation between the beams.
        direction = np.array([0.5, 1, 2])
        alpha = np.pi / 27 * 7
        force_fac = 0.01

        # Create mesh.
        for i in range(3):
            mesh = Mesh()

            # Create the first line.
            set_1 = create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat,
                [0, 0, 0], 1. * direction, n_el=3)

            if not i == 0:
                # In the second case rotate the line, so the triads do not
                # match any more.
                mesh.rotate(Rotation(direction, alpha))

            if i == 2:
                # The third line is with couplings.
                start_node = None
            else:
                start_node = set_1['end']

            # Add the second line.
            set_2 = create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat,
                1. * direction,
                2. * direction,
                n_el=3, start_node=start_node
                )

            # Add boundary conditions.
            mesh.add(BoundaryCondition(set_1['start'],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.dirichlet))
            mesh.add(BoundaryCondition(set_2['end'],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '{1} {1} {1} {1} {1} {1} 0 0 0 FUNCT {0} {0} {0} {0} {0} {0}'
                + ' 0 0 0',
                bc_type=mpy.neumann,
                format_replacement=[ft, force_fac]))

            if i == 2:
                # In the third case add a coupling.
                mesh.couple_nodes()

            # Add the mesh to the input file.
            input_file.add(mesh)

            # Each time move the whole mesh.
            input_file.translate([1, 0, 0])

        # Add result checks.
        displacement = [1.5015284845, 0.35139255451, -1.0126517891]
        nodes = [13, 26, 40]
        for node in nodes:
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[i]
                        )
                    )
                )

        # Run the input file in Baci.
        self.run_baci_test('rotated_beam_axis', input_file)


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
