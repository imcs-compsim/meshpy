# -*- coding: utf-8 -*-
"""
This script is used to test the functionality of the meshpy module.
"""

# Python imports.
import os
import unittest
import numpy as np
import autograd.numpy as npAD
import random
import warnings
import vtk

# Meshpy imports.
from meshpy import (mpy, Rotation, InputFile, MaterialReissner, MaterialBeam,
    BoundaryCondition, MaterialKirchhoff, Mesh, Coupling, Beam3rHerm2Line3,
    Function, MaterialEulerBernoulli, Beam3eb, InputSection, Beam3k,
    BaseMeshItem, set_header_static, set_beam_to_solid_meshtying,
    set_runtime_output, Beam3rLine2Line2)
from meshpy.node import Node
from meshpy.vtk_writer import VTKWriter
from meshpy.geometry_set import GeometrySet
from meshpy.container import GeometryName
from meshpy.element_beam import Beam
from meshpy.utility import (find_close_points, flatten,
    get_min_max_coordinates, partner_indices_to_point_partners,
    point_partners_to_partner_indices)

# Geometry functions.
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_line, create_beam_mesh_arc_segment)
from meshpy.mesh_creation_functions.beam_honeycomb import (
    create_beam_mesh_honeycomb, create_beam_mesh_honeycomb_flat)
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve

# Testing imports.
from tests.testing_utility import (testing_temp, testing_input,
    compare_strings, compare_vtk)


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
    beam = Beam3rHerm2Line3(material=material, nodes=mesh.nodes)
    mesh.add(beam)

    # Add a beam line with three elements
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, material,
        [100 * random.uniform(-1, 1) for _i in range(3)],
        [100 * random.uniform(-1, 1) for _i in range(3)],
        n_el=3)


class TestMeshpy(unittest.TestCase):
    """Test various stuff from the meshpy module."""

    def setUp(self):
        """
        This method is called before each test and sets the default meshpy
        values for each test. The values can be changed in the individual
        tests.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

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
            [100 * random.uniform(-1, 1) for _i in range(3)],
            100 * random.uniform(-1, 1)
            )
        origin = [100 * random.uniform(-1, 1) for _i in range(3)]

        for node in mesh_1.nodes:
            node.rotate(rot, origin=origin)

        mesh_2.rotate(rot, origin=origin)

        # Compare the output for the two meshes.
        compare_strings(self,
            'test_meshpy_rotate_mesh',
            mesh_1.get_string(header=False),
            mesh_2.get_string(header=False))

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
        compare_strings(self,
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
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
                    [0, 0, 0], [1, 0, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
                    [1, 0, 0], [1, 1, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
                    [1, 1, 0], [1, 1, 1], n_el=1)
            else:
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
                    [1, 0, 0], [0, 0, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
                    [1, 1, 0], [1, 0, 0], n_el=1)
                create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
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
            create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0],
                [-1, 0, 0], n_el=1)
            create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [-1, 0, 0],
                [-1, 1, 0], n_el=1)
            create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [-1, 1, 0],
                [-1, 1, 1], n_el=1)
            mesh.rotate(rot_1.inv())

            # Rotate everything, to show generalized reflection.
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
            compare_strings(self,
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

        # Convert the solid mesh to meshpy objects.
        mpy.import_mesh_full = full_import

        solid_file = os.path.join(testing_input,
            'test_meshpy_comments_in_input_file.dat')
        mesh = InputFile(dat_file=solid_file)

        # Add one element with BCs.
        mat = BaseMeshItem('material')
        sets = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0], [1, 2, 3])
        mesh.add(BoundaryCondition(sets['start'], 'test',
            bc_type=mpy.bc.dirichlet))
        mesh.add(BoundaryCondition(sets['end'], 'test',
            bc_type=mpy.bc.neumann))

        # Compare the output of the mesh.
        if full_import:
            full_name = 'create_comments_in_solid_full'
        else:
            full_name = 'create_comments_in_solid'
        compare_strings(self,
            full_name,
            ref_file,
            mesh.get_string(header=False).strip())

    def test_mesh_translations_with_solid(self):
        """Create a line that will be wrapped to a helix."""

        def base_test_mesh_translations(*, import_full=False, radius=None):
            """
            Create the line and wrap it with passing radius to the wrap
            function.
            """

            # Set default values for global parameters.
            mpy.set_default_values()

            # Convert the solid mesh to meshpy objects.
            mpy.import_mesh_full = import_full

            # Create the mesh.
            mesh = InputFile(dat_file=os.path.join(testing_input,
                'baci_input_solid_cuboid.dat'))
            mat = MaterialReissner(radius=0.05)

            # Create the line.
            create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
                [0.2, 0, 0],
                [0.2, 5 * 0.2 * 2 * np.pi, 4],
                n_el=3)

            # Transform the mesh.
            mesh.wrap_around_cylinder(radius=radius)
            mesh.translate([1, 2, 3])
            mesh.rotate(Rotation([1, 2, 3], np.pi * 17. / 27.))
            mesh.reflect([0.1, -2, 1])

            # Check the output.
            ref_file = os.path.join(testing_input,
                'test_meshpy_mesh_translations_with_solid_reference.dat')
            compare_strings(self,
                'test_meshpy_mesh_translations_with_solid',
                ref_file,
                mesh.get_string(header=False))

        base_test_mesh_translations(import_full=False, radius=None)
        base_test_mesh_translations(import_full=False, radius=0.2)
        self.assertRaises(ValueError,
            base_test_mesh_translations, import_full=False, radius=666)
        base_test_mesh_translations(import_full=True, radius=None)
        base_test_mesh_translations(import_full=True, radius=0.2)
        self.assertRaises(ValueError,
            base_test_mesh_translations, import_full=True, radius=666)

    def test_using_fluid_element_section(self):
        """"Add beam elements to an input file containing fluid elements"""

        input_file = InputFile(dat_file=os.path.join(testing_input,
            'fluid_element_input.dat'))

        beam_mesh = Mesh()
        material = MaterialEulerBernoulli(
        youngs_modulus=1e8,
        radius=0.001,
        density=10)
        beam_mesh.add(material)

        create_beam_mesh_line(beam_mesh,
            Beam3eb, material, [0, -0.5, 0], [0, 0.2, 0], n_el=5)
        input_file.add(beam_mesh)

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_mesh_beam_with_fluid_reference.dat')
        compare_strings(self,
            'test_meshpy_mesh_beam_with_fluid',
            ref_file,
            input_file.get_string(header=False))


    def test_wrap_cylinder_not_on_same_plane(self):
        """Create a helix that is itself wrapped around a cylinder."""

        # Ignore the warnings from wrap around cylinder.
        warnings.filterwarnings("ignore")

        # Create the mesh.
        mesh = InputFile()
        mat = MaterialReissner(radius=0.05)

        # Create the line and bend it to a helix.
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0.2, 0, 0],
            [0.2, 5 * 0.2 * 2 * np.pi, 4],
            n_el=20)
        mesh.wrap_around_cylinder()

        # Move the helix so its axis is in the y direction and goes through
        # (2 0 0). The helix is also moved by a lot in y-direction, this only
        # affects the angle phi when wrapping around a cylinder, not the shape
        # of the beam.
        mesh.rotate(Rotation([1, 0, 0], -0.5 * np.pi))
        mesh.translate([2, 666.666, 0])

        # Wrap the helix again.
        mesh.wrap_around_cylinder(radius=2.)

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_wrap_cylinder_not_on_same_plane_reference.dat')
        compare_strings(self,
            'test_meshpy_wrap_cylinder_not_on_same_plane',
            ref_file,
            mesh.get_string(header=False))

    def test_get_nodes_by_function(self):
        """
        Check if the get_nodes_by_function method of Mesh works properly.
        """

        def get_nodes_at_x(node, x_value):
            """True for all coordinates at a certain x value."""
            if np.abs(node.coordinates[0] - x_value) < 1e-10:
                return True
            else:
                return False

        mat = MaterialReissner()

        mesh = Mesh()
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0],
            [5, 0, 0],
            n_el=5)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 1, 0],
            [10, 1, 0],
            n_el=10)

        # Add a dummy node to check that dat file nodes are skipped.
        mesh.add_node(BaseMeshItem())

        nodes = mesh.get_nodes_by_function(get_nodes_at_x, 1.0)
        self.assertTrue(2 == len(nodes))
        for node in nodes:
            self.assertTrue(np.abs(1.0 - node.coordinates[0]) < 1e-10)

    def test_get_min_max_coordinates(self):
        """
        Test if the get_min_max_coordinates function works properly.
        """

        # Create the mesh.
        mpy.import_mesh_full = True
        mesh = InputFile(dat_file=os.path.join(testing_input,
            'baci_input_solid_cuboid.dat'))
        mat = MaterialReissner(radius=0.05)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0],
            [2, 3, 4],
            n_el=10)

        # Check the results.
        min_max = get_min_max_coordinates(mesh.nodes)
        ref_solution = [-0.5, -1.0, -1.5, 2.0, 3.0, 4.0]
        self.assertTrue(np.linalg.norm(min_max - ref_solution) < 1e-10)

    def test_find_close_points_binning(self):
        """
        Test if the find_close_points_binning and find_close_points functions
        return the same results.
        """

        # Set the seed for the pseudo random numbers.
        random.seed(0)

        # Add random nodes to a cube with width 2. Randomly add nodes close to
        # existing nodes. The distance has to be shorter than 0.5 * eps_medium
        # since the algorithm needs all close nodes to be within a sphere with
        # radius eps_medium.
        eps_medium = 1e-5
        eps_medium_factor = 0.49 * eps_medium
        n_nodes = 999
        coords = np.zeros([n_nodes, 3])
        for i in range(n_nodes):
            # Check if this one should be close to another one.
            if random.randint(0, 4) == 0 and i > 0:
                close_node = random.randint(0, i - 1)
                for j in range(3):
                    coords[i, j] = coords[close_node, j]
            else:
                for j in range(3):
                    coords[i, j] = random.uniform(-1, 1)

        # Create a random vector for each node. The length of the random
        # vectors is scaled, so that is is a maximum of 1.
        diff = np.random.rand(n_nodes, 3)
        diff /= np.linalg.norm(diff, axis=1)[:, None]
        coords += eps_medium_factor * diff

        # Add nodes between the bins.
        coords[-1, :] = [0., 0., 0.]
        coords[-2, :] = [0., 0., 0.]
        coords[-3, :] = [0., 0., 0.5]
        coords[-4, :] = [0., 0., 0.5]
        coords[-5, :] = [0., 0.5, 0.]
        coords[-6, :] = [0., 0.5, 0.]
        coords[-7, :] = [0.5, 0., 0.]
        coords[-8, :] = [0.5, 0., 0.]
        coords[-9, :] = [0. + eps_medium_factor, 0., 0.]
        coords[-10, :] = [0., 0. + eps_medium_factor, 0.]
        coords[-11, :] = [0., 0., 0. + eps_medium_factor]
        coords[-12, :] = [0. - eps_medium_factor, 0., 0.]
        coords[-13, :] = [0., 0. - eps_medium_factor, 0.]
        coords[-14, :] = [0., 0., 0. - eps_medium_factor]

        # Test the number of partners and list of partners with the different
        # methods.
        partner_indices = find_close_points(coords, binning=True, nx=4,
            ny=4, nz=4, eps=eps_medium)
        partner_indices_brute = find_close_points(coords, binning=False,
            eps=eps_medium)
        self.assertTrue(np.array_equal(partner_indices, partner_indices_brute))
        self.assertEqual(len(partner_indices), 146)

        # Also compare to the reference solution.
        point_partners_reference = -1 * np.ones([n_nodes], dtype=int)
        index_vector = [0, 1, 3, 6, 8, 9, 10, 11, 12, 13, 16, 19, 20, 21, 22,
            25, 26, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 43, 44, 48, 49,
            50, 51, 55, 59, 61, 62, 63, 65, 66, 67, 68, 69, 72, 73, 74, 77, 78,
            79, 80, 82, 83, 84, 85, 86, 88, 89, 92, 94, 95, 96, 98, 103, 104,
            106, 110, 111, 115, 119, 120, 121, 123, 125, 127, 128, 130, 132,
            133, 134, 136, 137, 138, 139, 140, 141, 143, 144, 146, 147, 149,
            151, 154, 156, 157, 160, 165, 168, 170, 172, 175, 176, 179, 180,
            181, 183, 185, 188, 189, 190, 193, 196, 197, 198, 202, 203, 206,
            209, 211, 212, 214, 215, 219, 221, 222, 225, 229, 230, 232, 239,
            241, 242, 245, 247, 248, 250, 253, 254, 255, 256, 260, 261, 265,
            270, 273, 276, 278, 281, 288, 289, 295, 301, 302, 305, 308, 312,
            313, 315, 318, 322, 328, 337, 343, 346, 347, 352, 354, 356, 359,
            361, 363, 365, 367, 368, 369, 373, 376, 377, 379, 381, 384, 391,
            392, 393, 397, 398, 400, 401, 405, 417, 419, 421, 424, 429, 431,
            435, 437, 440, 441, 442, 444, 446, 456, 459, 462, 469, 470, 471,
            473, 474, 475, 476, 480, 484, 486, 490, 498, 501, 506, 513, 518,
            522, 523, 524, 525, 526, 536, 539, 541, 544, 545, 548, 560, 567,
            570, 571, 572, 576, 577, 596, 598, 600, 601, 603, 606, 611, 615,
            616, 621, 622, 625, 632, 633, 637, 639, 643, 646, 647, 649, 650,
            653, 672, 675, 676, 678, 679, 680, 682, 691, 693, 695, 700, 701,
            715, 719, 721, 725, 730, 736, 738, 742, 746, 749, 750, 756, 759,
            761, 762, 767, 769, 777, 778, 781, 783, 785, 787, 789, 792, 795,
            806, 811, 817, 826, 827, 830, 833, 834, 835, 839, 840, 841, 842,
            847, 850, 853, 854, 857, 859, 861, 864, 867, 869, 872, 875, 887,
            890, 895, 902, 904, 905, 912, 913, 914, 915, 926, 928, 932, 934,
            935, 944, 956, 958, 965, 979, 985, 986, 987, 988, 989, 990, 991,
            992, 993, 994, 995, 996, 997, 998]
        val_vector = [0, 1, 2, 3, 4, 5, 5, 6, 2, 7, 8, 9, 9, 10, 11, 12, 13,
            11, 14, 2, 15, 16, 17, 18, 19, 20, 15, 21, 22, 22, 23, 11, 10, 16,
            0, 5, 24, 2, 25, 8, 16, 26, 26, 27, 15, 28, 9, 25, 29, 30, 31, 32,
            33, 20, 34, 35, 36, 19, 5, 37, 38, 39, 26, 40, 41, 42, 16, 4, 43,
            44, 37, 45, 46, 24, 35, 47, 48, 49, 44, 50, 51, 49, 52, 53, 5, 54,
            55, 18, 42, 12, 56, 57, 1, 58, 59, 51, 60, 39, 61, 62, 63, 64, 65,
            27, 0, 66, 67, 68, 69, 70, 71, 50, 63, 63, 72, 73, 74, 75, 76, 77,
            6, 21, 73, 44, 23, 78, 24, 76, 75, 22, 69, 79, 80, 54, 81, 16, 82,
            59, 50, 83, 84, 85, 3, 75, 86, 35, 87, 66, 53, 48, 13, 88, 31, 73,
            85, 67, 89, 90, 39, 47, 17, 21, 26, 87, 3, 62, 91, 78, 75, 7, 60,
            92, 61, 93, 94, 54, 95, 96, 60, 97, 98, 86, 28, 99, 68, 100, 101,
            56, 100, 102, 67, 103, 104, 80, 105, 57, 103, 79, 106, 46, 82, 58,
            107, 69, 50, 108, 90, 109, 47, 92, 53, 91, 84, 110, 111, 30, 112,
            106, 98, 113, 114, 82, 115, 45, 116, 62, 117, 34, 35, 118, 32, 4,
            39, 109, 32, 33, 81, 115, 107, 10, 119, 64, 118, 120, 52, 108, 119,
            70, 102, 121, 23, 117, 5, 97, 122, 123, 124, 99, 111, 95, 125, 83,
            126, 127, 123, 114, 126, 107, 128, 117, 129, 130, 65, 116, 75, 72,
            131, 93, 132, 40, 129, 133, 131, 18, 134, 95, 124, 135, 77, 58,
            105, 128, 38, 136, 21, 41, 130, 137, 120, 132, 36, 138, 127, 110,
            136, 22, 96, 88, 29, 121, 74, 112, 133, 58, 42, 139, 64, 113, 117,
            125, 135, 137, 63, 14, 59, 83, 138, 140, 141, 94, 21, 140, 122, 89,
            43, 71, 101, 1, 139, 55, 134, 63, 104, 141, 142, 142, 142, 142,
            142, 142, 143, 143, 144, 144, 145, 145, 142, 142]
        point_partners_reference[index_vector] = val_vector
        point_partners, _n_partners = partner_indices_to_point_partners(
            partner_indices, len(coords))
        self.assertTrue(np.array_equal(point_partners,
            point_partners_reference))

    def test_find_close_points_binning_flat(self):
        """
        Test case for coupling of points, when the nodes are all on a plane.
        """

        # Dummy material.
        material = MaterialReissner(radius=0.1)

        def create_flat_mesh():
            """Create a flat honeycomb mesh."""
            input_file = InputFile()
            create_beam_mesh_honeycomb_flat(input_file, Beam3rHerm2Line3,
                material, 1, 5, 5,
                create_couplings=False)
            return input_file

        # Get a reference solution for the coupling nodes.
        reference_partners_list = [0, 1, 1, 2, 0, 3, 2, 4, 4, 5, 2, 6, 5, 7, 7,
            8, 5, 9, 8, 10, 10, 11, 8, 12, 11, 13, 13, 14, 11, 15, 14, 16, 3,
            17, 17, 6, 17, 18, 6, 19, 19, 9, 19, 20, 9, 21, 21, 12, 21, 22, 12,
            23, 23, 15, 23, 24, 15, 25, 25, 16, 25, 26, 27, 18, 18, 28, 27, 29,
            28, 20, 20, 30, 28, 31, 30, 22, 22, 32, 30, 33, 32, 24, 24, 34, 32,
            35, 34, 26, 26, 36, 34, 37, 36, 38, 29, 39, 39, 31, 39, 40, 31, 41,
            41, 33, 41, 42, 33, 43, 43, 35, 43, 44, 35, 45, 45, 37, 45, 46, 37,
            47, 47, 38, 47, 48, 49, 40, 40, 50, 49, 51, 50, 42, 42, 52, 50, 53,
            52, 44, 44, 54, 52, 55, 54, 46, 46, 56, 54, 57, 56, 48, 48, 58, 56,
            59, 58, 60, 51, 61, 61, 53, 53, 62, 62, 55, 55, 63, 63, 57, 57, 64,
            64, 59, 59, 65, 65, 60]
        reference_partners = point_partners_to_partner_indices(
            reference_partners_list, 66)

        # The reference data was created for the nodes without the middle
        # nodes, therefore we filter the middle nodes here.
        partners_no_binning = find_close_points(
            create_flat_mesh().get_global_coordinates(middle_nodes=False)[0],
            binning=False)
        self.assertTrue(np.array_equal(reference_partners,
            partners_no_binning))

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
            partners_binning = find_close_points(
                input_file.get_global_coordinates(middle_nodes=False)[0],
                binning=True)

            # Compare the partners with the reference.
            self.assertTrue(np.array_equal(partners_binning,
                partners_no_binning))

    def test_find_close_points_dimension(self):
        """
        Test that the find_close_points function also works properly with
        multidimensional points.
        """

        # Set the seed for the pseudo random numbers.
        random.seed(0)

        # Create array with coordinates.
        coords = np.array([
            [0., 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [-1, 1, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 0],
            [-1, -1, 1, 0, 0, 0],
            [1, 1, -1, 0, 0, 0],
            [-1, 1, -1, 0, 0, 0],
            [1, -1, -1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0],
            [1, 1, 1, 0, 0, 1],
            [-1, 1, 1, 0, 0, 1],
            [1, -1, 1, 0, 0, 1],
            [-1, -1, 1, 0, 0, 0],
            [1, 1, -1, 0, 0, 0],
            [-1, 1, -1, 0, 0, 0],
            [1, -1, -1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0],
            ])

        # Expected results.
        has_partner_expected = [-1, -1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, 0,
            1, 2, 3, 4]
        partner_expected = 5

        # Get results with binning.
        partner_indices = find_close_points(coords, binning=True)
        has_partner_binning, partner_binning = (
            partner_indices_to_point_partners(partner_indices, len(coords)))

        # Get results without binning.
        partner_indices = find_close_points(coords, binning=False)
        has_partner_brute, partner_brute = (
            partner_indices_to_point_partners(partner_indices, len(coords)))

        # Check the results.
        self.assertTrue(has_partner_expected, has_partner_binning)
        self.assertTrue(has_partner_expected, has_partner_brute)
        self.assertEqual(partner_expected, partner_binning)
        self.assertEqual(partner_expected, partner_brute)

    def test_curve_3d_helix(self):
        """
        Create a helix from a parametric curve where the parameter is
        transformed so the arc length along the beam is not proportional to
        the parameter.
        """

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
        helix_set = create_beam_mesh_curve(input_file, Beam3rHerm2Line3, mat,
            helix, [0., 2. * np.pi * n], n_el=n_el)

        # Compare the coordinates with the ones from Mathematica.
        coordinates_mathematica = np.loadtxt(os.path.join(testing_input,
                'test_meshpy_curve_3d_helix_mathematica.csv'), delimiter=',')
        self.assertLess(
            np.linalg.norm(
                coordinates_mathematica
                - input_file.get_global_coordinates()[0]),
            mpy.eps_pos,
            'test_meshpy_curve_3d_helix'
            )

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(helix_set['start'], 'BC1',
            bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(helix_set['end'], 'BC2',
            bc_type=mpy.bc.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_3d_helix_reference.dat')
        compare_strings(self,
            'test_meshpy_curve_3d_helix',
            ref_file,
            input_file.get_string(header=False))

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
        sin_set = create_beam_mesh_curve(input_file, Beam3rHerm2Line3, mat,
            sin, [0., 2. * np.pi], n_el=n_el)

        # Compare the coordinates with the ones from Mathematica.
        coordinates_mathematica = np.loadtxt(os.path.join(testing_input,
                'test_meshpy_curve_2d_sin_mathematica.csv'), delimiter=',')
        self.assertLess(
            np.linalg.norm(
                coordinates_mathematica
                - input_file.get_global_coordinates()[0]),
            mpy.eps_pos,
            'test_meshpy_curve_2d_sin'
            )

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(sin_set['start'], 'BC1',
            bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(sin_set['end'], 'BC2',
            bc_type=mpy.bc.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_2d_sin_reference.dat')
        compare_strings(self,
            'test_meshpy_curve_2d_sin',
            ref_file,
            input_file.get_string(header=False))

    def test_curve_3d_curve_rotation(self):
        """Create a line from a parametric curve and prescribe the rotation."""

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
        sin_set = create_beam_mesh_curve(input_file, Beam3rHerm2Line3, mat,
            curve, [0., 1.], n_el=n_el, function_rotation=rotation)

        # Apply boundary conditions.
        input_file.add(BoundaryCondition(sin_set['start'], 'BC1',
            bc_type=mpy.bc.dirichlet))
        input_file.add(BoundaryCondition(sin_set['end'], 'BC2',
            bc_type=mpy.bc.neumann))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_3d_line_rotation_reference.dat')
        compare_strings(self,
            'test_meshpy_curve_3d_line_rotation',
            ref_file,
            input_file.get_string(header=False))

    def test_curve_3d_line(self):
        """
        Create a line from a parametric curve. Once the interval is in
        ascending order, once in descending. This tests checks that the
        elements are created with the correct tangent vectors.
        """

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

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
        set_1 = create_beam_mesh_curve(input_file, Beam3rHerm2Line3, mat, line,
            [0., 5.], n_el=3)
        input_file.translate([0, 1, 0])
        set_2 = create_beam_mesh_curve(input_file, Beam3rHerm2Line3, mat, line,
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
                bc_type=mpy.bc.dirichlet))
            input_file.add(BoundaryCondition(set_item[name_end],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '{1} {1} {1} {1} {1} {1} 0 0 0 FUNCT {0} {0} {0} {0} {0} {0}'
                + ' 0 0 0',
                bc_type=mpy.bc.neumann,
                format_replacement=[ft, force_fac]))

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_curve_3d_line_reference.dat')
        compare_strings(self,
            'test_meshpy_curve_3d_line',
            ref_file,
            input_file.get_string(header=False))

    def test_reissner_beam(self):
        """
        Test that the input file for all types of Reissner beams is generated
        correctly.
        """

        # Create input file.
        material = MaterialReissner(radius=0.1, youngs_modulus=1000)
        input_file = InputFile()

        # Create a beam arc with the different Reissner beam types.
        for i, beam_type in enumerate([Beam3rHerm2Line3, Beam3rLine2Line2]):
            create_beam_mesh_arc_segment(input_file, beam_type, material,
                [0.0, 0.0, i], Rotation([0.0, 0.0, 1.0], np.pi / 2.0), 2.0,
                np.pi / 2.0, n_el=2)

        # Compare with the reference solution.
        ref_file = os.path.join(testing_input,
            'test_meshpy_reissner_beam_reference.dat')
        compare_strings(self,
            'test_reissner_beam',
            ref_file,
            input_file.get_string(header=False))

    def test_kirchhoff_beam(self):
        """
        Test that the input file for all types of Kirchhoff beams is generated
        correctly.
        """

        # Create input file.
        material = MaterialKirchhoff(radius=0.1, youngs_modulus=1000)
        input_file = InputFile()

        with warnings.catch_warnings():
            # Ignore the warnings for the rotvec beams.
            warnings.simplefilter("ignore")

            # Loop over options.
            for FAD in [True, False]:
                for weak in [True, False]:
                    for rotvec in [True, False]:
                        # Define the beam object factory function for the
                        # creation functions.
                        BeamObject = Beam3k(weak=weak, rotvec=rotvec, FAD=FAD)

                        # Create a beam.
                        set_1 = create_beam_mesh_line(input_file, BeamObject,
                            material, [0, 0, 0], [1, 0, 0], n_el=2)
                        set_2 = create_beam_mesh_line(input_file, BeamObject,
                            material, [1, 0, 0], [2, 0, 0], n_el=2)

                        # Couple the nodes.
                        if rotvec:
                            input_file.couple_nodes(nodes=flatten(
                                [set_1['end'].nodes, set_2['start'].nodes]))

                        # Move the mesh away from the next created beam.
                        input_file.translate([0, 0.5, 0])

        # Compare with the reference solution.
        ref_file = os.path.join(testing_input,
            'test_meshpy_kirchhoff_beam_reference.dat')
        compare_strings(self,
            'test_kirchhoff_beam',
            ref_file,
            input_file.get_string(header=False))

    def test_euler_bernoulli_beam(self):
        """
        Recreate the baci test case beam3eb_static_endmoment_quartercircle.dat
        This tests the implementation for Euler Bernoulli beams.
        """

        # Create the input file and add function and material.
        input_file = InputFile()
        fun = Function('COMPONENT 0 FUNCTION t')
        input_file.add(fun)
        mat = MaterialEulerBernoulli(youngs_modulus=1., density=1.3e9)

        # Set the parameters that are also set in the test file.
        mat.area = 1
        mat.mom2 = 1e-4

        # Create the beam.
        beam_set = create_beam_mesh_line(input_file, Beam3eb, mat,
            [-1, 0, 0], [1, 0, 0], n_el=16)

        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(
                beam_set['start'],
                'NUMDOF 6 ONOFF 1 1 1 0 1 1 VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                + 'FUNCT 0 0 0 0 0 0',
                bc_type=mpy.bc.dirichlet
                )
            )
        input_file.add(
            BoundaryCondition(
                beam_set['end'],
                'NUMDOF 6 ONOFF 0 0 0 0 0 1 VAL 0.0 0.0 0.0 0.0 0.0 '
                + '7.8539816339744e-05 FUNCT 0 0 0 0 0 {}',
                bc_type=mpy.bc.moment_euler_bernoulli, format_replacement=[fun]
                )
            )

        # Compare with the reference solution.
        ref_file = os.path.join(testing_input,
            'test_meshpy_euler_bernoulli_reference.dat')
        compare_strings(self,
            'test_meshpy_euler_bernoulli',
            ref_file,
            input_file.get_string(header=False, check_nox=False))

        # Test consistency checks.
        rot = Rotation([1, 2, 3], 2.3434)
        input_file.nodes[-1].rotation = rot
        with self.assertRaises(ValueError):
            # This raises an error because not all rotation in the beams are
            # the same.
            input_file.get_string(header=False)

        for node in input_file.nodes:
            node.rotation = rot
        with self.assertRaises(ValueError):
            # This raises an error because the rotations do not match the
            # director between the nodes.
            input_file.get_string(header=False)

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
                element = Beam3rHerm2Line3(mat, nodes)
                input_file.add(element)

            # Add sets.
            geom_set = GeometryName()
            geom_set['start'] = GeometrySet(mpy.geo.point,
                nodes=input_file.nodes[0])
            geom_set['end'] = GeometrySet(mpy.geo.point,
                nodes=input_file.nodes[0])
            geom_set['line'] = GeometrySet(mpy.geo.line,
                nodes=input_file.nodes)
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
            geom_set['start'] = GeometrySet(mpy.geo.point,
                nodes=set_1['start'])
            geom_set['end'] = GeometrySet(mpy.geo.point, nodes=set_2['end'])
            geom_set['line'] = GeometrySet(mpy.geo.line,
                nodes=[set_1['line'], set_2['line']],
                fail_on_double_nodes=False)
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
                'beam_object': Beam3rHerm2Line3,
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
                'beam_object': Beam3rHerm2Line3,
                'material': mat,
                'function': circle_function,
                'interval': arg_interval,
                'n_el': arg_n_el
                }

        # Check the meshes without additional rotation.
        ref_file = os.path.join(testing_input,
            'test_meshpy_close_beam_reference.dat')

        compare_strings(self,
            'test_meshpy_close_beam_manual',
            ref_file,
            create_mesh_manually(Rotation()).get_string(header=False))
        compare_strings(self,
            'test_meshpy_close_beam_full_segment',
            ref_file,
            one_full_circle_closed(create_beam_mesh_arc_segment,
                get_arguments_arc_segment(0)).get_string(header=False))
        compare_strings(self,
            'test_meshpy_close_beam_split_segment',
            ref_file,
            two_half_circles_closed(create_beam_mesh_arc_segment,
                [get_arguments_arc_segment(1), get_arguments_arc_segment(2)]
                ).get_string(header=False))
        compare_strings(self,
            'test_meshpy_close_beam_full_curve',
            ref_file,
            one_full_circle_closed(create_beam_mesh_curve,
                get_arguments_curve(0)).get_string(header=False))
        compare_strings(self,
            'test_meshpy_close_beam_split_curve',
            ref_file,
            two_half_circles_closed(create_beam_mesh_curve,
                [get_arguments_curve(1), get_arguments_curve(2)]
                ).get_string(header=False))

        # Check the meshes with additional rotation.
        ref_file = os.path.join(testing_input,
            'test_meshpy_close_beam_rotated_reference.dat')

        compare_strings(self,
            'test_meshpy_close_beam_manual_rotation',
            ref_file,
            create_mesh_manually(additional_rotation).get_string(
                header=False))
        compare_strings(self,
            'test_meshpy_close_beam_full_segment_rotation',
            ref_file,
            one_full_circle_closed(create_beam_mesh_arc_segment,
                get_arguments_arc_segment(0),
                additional_rotation=additional_rotation).get_string(
                    header=False))
        compare_strings(self,
            'test_meshpy_close_beam_split_segment_rotation',
            ref_file,
            two_half_circles_closed(create_beam_mesh_arc_segment,
                [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
                additional_rotation=additional_rotation
                ).get_string(header=False))
        compare_strings(self,
            'test_meshpy_close_beam_full_curve_rotation',
            ref_file,
            one_full_circle_closed(create_beam_mesh_curve,
                get_arguments_curve(0),
                additional_rotation=additional_rotation).get_string(
                    header=False))
        compare_strings(self,
            'test_meshpy_close_beam_split_curve_rotation',
            ref_file,
            two_half_circles_closed(create_beam_mesh_curve,
                [get_arguments_curve(1), get_arguments_curve(2)],
                additional_rotation=additional_rotation
                ).get_string(header=False))

    def test_replace_nodes(self):
        """Test case for coupling of nodes, and reusing the identical nodes."""

        mpy.check_overlapping_elements = False

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
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0],
            [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0],
            [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0],
            [2, 0, 0])

        ref_nodes = list(mesh_ref.nodes)
        coupling_nodes = list(mesh_couple.nodes)

        # Add a line set with all nodes, to check that the nodes in the
        # boundary condition are replaced correctly.
        mesh_ref.add(GeometrySet(mpy.geo.line, nodes=ref_nodes))
        mesh_couple.add(GeometrySet(mpy.geo.line, nodes=coupling_nodes))

        # Add another line set with all nodes, this time only the coupling node
        # that will be kept is in this set.
        mesh_ref.add(GeometrySet(mpy.geo.line, nodes=ref_nodes))
        coupling_nodes_without_replace_node = list(coupling_nodes)
        del coupling_nodes_without_replace_node[3]
        mesh_couple.add(GeometrySet(mpy.geo.line,
            nodes=coupling_nodes_without_replace_node))

        # Add another line set with all nodes, this time only the coupling node
        # that will be replaced is in this set.
        mesh_ref.add(GeometrySet(mpy.geo.line, nodes=ref_nodes))
        coupling_nodes_without_replace_node = list(coupling_nodes)
        del coupling_nodes_without_replace_node[2]
        mesh_couple.add(GeometrySet(mpy.geo.line,
            nodes=coupling_nodes_without_replace_node))

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the coupling mesh.
        mesh_couple.couple_nodes(coupling_type=mpy.coupling.fix_reuse)

        # Compare the meshes.
        compare_strings(self, 'test_replace_nodes_case_1',
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False))

        # Create two overlapping beams. This is to test that the middle nodes
        # are not coupled.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        set_ref = create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat,
            [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0],
            [1, 0, 0], start_node=set_ref['start'], end_node=set_ref['end'])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0],
            [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0],
            [1, 0, 0])

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the coupling mesh.
        mesh_couple.couple_nodes(coupling_type=mpy.coupling.fix_reuse)

        # Compare the meshes.
        compare_strings(self, 'test_replace_nodes_case_2',
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False))

        # Create a beam with two elements. Once immediately and once as two
        # beams with couplings.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0],
            [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0],
            [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0],
            [2, 0, 0])

        # Create set with all the beam nodes.
        node_set_1_ref = GeometrySet(mpy.geo.line, nodes=mesh_ref.nodes)
        node_set_2_ref = GeometrySet(mpy.geo.line, nodes=mesh_ref.nodes)
        node_set_1_couple = GeometrySet(mpy.geo.line, nodes=mesh_couple.nodes)
        node_set_2_couple = GeometrySet(mpy.geo.line, nodes=mesh_couple.nodes)

        # Create connecting beams.
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0],
            [2, 2, 2])
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0],
            [2, -2, -2])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0],
            [2, 2, 2])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0],
            [2, -2, -2])

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the mesh.
        mesh_ref.couple_nodes(coupling_type=mpy.coupling.fix)
        mesh_couple.couple_nodes(coupling_type=mpy.coupling.fix_reuse)

        # Add the node sets.
        mesh_ref.add(node_set_1_ref)
        mesh_couple.add(node_set_1_couple)

        # Add BCs.
        mesh_ref.add(BoundaryCondition(node_set_2_ref, 'BC1',
            bc_type=mpy.bc.neumann))
        mesh_couple.add(BoundaryCondition(node_set_2_couple, 'BC1',
            bc_type=mpy.bc.neumann))

        # Compare the meshes.
        compare_strings(self, 'test_replace_nodes_case_3',
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False))

    def create_beam_to_solid_conditions_model(self):
        """
        Create the inpuf file for the beam-to-solid input conditions tests.
        """

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher',
            dat_file=os.path.join(testing_input,
                'test_meshpy_btsvm_coupling_solid_mesh.dat'))

        # Add beams to the model.
        beam_mesh = Mesh()
        material = MaterialReissner(youngs_modulus=1000, radius=0.05)
        create_beam_mesh_line(beam_mesh, Beam3rHerm2Line3,
            material, [0, 0, 0], [0, 0, 1], n_el=3)
        create_beam_mesh_line(beam_mesh, Beam3rHerm2Line3,
            material, [0, 0.5, 0], [0, 0.5, 1], n_el=3)

        # Set beam-to-solid coupling conditions.
        line_set = GeometrySet(mpy.geo.line, beam_mesh.nodes)
        beam_mesh.add(
            BoundaryCondition(
                line_set,
                bc_type=mpy.bc.beam_to_solid_volume_meshtying,
                bc_string='COUPLING_ID 1'
                )
            )
        beam_mesh.add(
            BoundaryCondition(
                line_set,
                bc_type=mpy.bc.beam_to_solid_surface_meshtying,
                bc_string='COUPLING_ID 2'
                )
            )

        # Add the beam to the solid mesh.
        input_file.add(beam_mesh)
        return input_file

    def test_beam_to_solid_conditions(self):
        """
        Create beam-to-solid input conditions.
        """

        # Create input file.
        input_file = self.create_beam_to_solid_conditions_model()

        # Compare with the reference file.
        compare_strings(self, 'test_meshpy_btsvm_coupling',
            os.path.join(testing_input,
                'test_meshpy_btsvm_coupling_reference.dat'),
            input_file.get_string(header=False))

    def test_beam_to_solid_conditions_full(self):
        """
        Create beam-to-solid input conditions with full import.
        """

        # Create input file.
        mpy.import_mesh_full = True
        input_file = self.create_beam_to_solid_conditions_model()

        # Compare with the reference file.
        compare_strings(self, 'test_meshpy_btsvm_coupling_full',
            os.path.join(testing_input,
                'test_meshpy_btsvm_coupling_full_reference.dat'),
            input_file.get_string(header=False))

    def test_nurbs_import(self):
        """
        Test if the import of a nurbs mesh works as expected.
        This script generates the baci test case:
        beam3r_herm2line3_static_beam_to_solid_volume_meshtying_nurbs27_mortar_penalty_line4
        """

        # Create beam mesh and load solid file.
        input_file = InputFile('Ivo Steinbrecher',
            dat_file=os.path.join(
                testing_input,
                'test_meshpy_nurbs_import_solid_mesh.dat')
            )
        set_header_static(input_file, time_step=0.5, n_steps=2,
            tol_residuum=1e-14, tol_increment=1e-8,
            option_overwrite=True)
        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.volume_meshtying,
            contact_discretization='mortar',
            mortar_shape='line4',
            penalty_parameter=1000,
            n_gauss_points=6,
            segmentation=True,
            binning_bounding_box=[-3, -3, -1, 3, 3, 5],
            binning_cutoff_radius=1)
        set_runtime_output(input_file, output_solid=False)
        input_file.add(InputSection('IO',
            '''
            OUTPUT_BIN     yes
            STRUCT_DISP    yes
            FILESTEPS      1000
            VERBOSITY      Standard
            ''',
            option_overwrite=True))
        fun = Function('COMPONENT 0 FUNCTION t')
        input_file.add(fun)

        # Create the beam material.
        material = MaterialReissner(youngs_modulus=1000, radius=0.05)

        # Create the beams.
        set_1 = create_beam_mesh_line(input_file, Beam3rHerm2Line3, material,
            [0, 0, 0.95], [1, 0, 0.95], n_el=2)
        set_2 = create_beam_mesh_line(input_file, Beam3rHerm2Line3, material,
            [-0.25, -0.3, 0.85], [-0.25, 0.5, 0.85], n_el=2)

        # Add boundary conditions on the beams.
        input_file.add(
            BoundaryCondition(set_1['start'],
                'NUMDOF 9 ONOFF 0 0 0 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 '
                + 'FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.bc.dirichlet))
        input_file.add(
            BoundaryCondition(set_1['end'],
                'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 0.02 0 0 0 0 0 0 0 '
                + 'FUNCT 0 {} 0 0 0 0 0 0 0',
                format_replacement=[fun],
                bc_type=mpy.bc.neumann))
        input_file.add(
            BoundaryCondition(set_2['start'],
                'NUMDOF 9 ONOFF 0 0 0 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 '
                + 'FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.bc.dirichlet))
        input_file.add(
            BoundaryCondition(set_2['end'],
                'NUMDOF 9 ONOFF 1 0 0 0 0 0 0 0 0 VAL -0.06 0 0 0 0 0 0 0 0 '
                + 'FUNCT {} 0 0 0 0 0 0 0 0',
                format_replacement=[fun],
                bc_type=mpy.bc.neumann))

        # Add result checks.
        displacement = [
            [-5.14451531793581718e-01, -1.05846397858073843e-01,
                -1.77822866851472888e-01]
            ]

        nodes = [64]
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

        # Compare with the reference solution.
        ref_file = os.path.join(testing_input,
            'test_meshpy_nurbs_import_reference.dat')
        compare_strings(self,
            'test_meshpy_nurbs_import',
            ref_file,
            input_file.get_string(header=False, check_nox=False))

    def test_vtk_writer(self):
        """Test the output created by the VTK writer."""

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

        # Add tetrahedron.
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

        # Compare the vtk files.
        compare_vtk(self, 'test_meshpy_vtk_writer', ref_file, vtk_file)

    def test_vtk_writer_beam(self):
        """Create a sample mesh and check the VTK output."""

        # Create the mesh.
        mesh = Mesh()

        # Add content to the mesh.
        mat = MaterialBeam(radius=0.05)
        create_beam_mesh_honeycomb(mesh, Beam3rHerm2Line3, mat, 2., 2, 3,
            n_el=2, add_sets=True)

        # Write VTK output, with coupling sets."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_beam_reference.vtu')
        vtk_file = os.path.join(testing_temp, 'test_meshpy_vtk_beam.vtu')
        mesh.write_vtk(output_name='test_meshpy_vtk', coupling_sets=True,
            output_directory=testing_temp, ascii=True)

        # Compare the vtk files.
        compare_vtk(self, 'test_vtk_beam', ref_file, vtk_file,
            tol_float=mpy.eps_pos)

        # Write VTK output, without coupling sets."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_no_coupling_beam_reference.vtu')
        vtk_file = os.path.join(testing_temp,
            'test_meshpy_vtk_no_coupling_beam.vtu')
        mesh.write_vtk(output_name='test_meshpy_vtk_no_coupling',
            coupling_sets=False, output_directory=testing_temp, ascii=True)

        # Compare the vtk files.
        compare_vtk(self, 'test_meshpy_vtk_no_coupling_beam', ref_file,
            vtk_file, tol_float=mpy.eps_pos)

    def test_vtk_writer_solid(self):
        """Import a solid mesh and check the VTK output."""

        # Convert the solid mesh to meshpy objects. Without this parameter no
        # solid VTK file would be written.
        mpy.import_mesh_full = True

        # Create the input file and read solid mesh data.
        input_file = InputFile()
        input_file.read_dat(os.path.join(testing_input,
            'baci_input_solid_tube.dat'))

        # Write VTK output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_solid_reference.vtu')
        vtk_file = os.path.join(testing_temp, 'test_meshpy_vtk_solid.vtu')
        if os.path.isfile(vtk_file):
            os.remove(vtk_file)
        input_file.write_vtk(output_name='test_meshpy_vtk',
            output_directory=testing_temp, ascii=True)

        # Compare the vtk files.
        compare_vtk(self, 'test_meshpy_vtk_solid', ref_file, vtk_file)

    def test_vtk_writer_solid_elements(self):
        """
        Import a solid mesh with all solid types and check the VTK output.
        """

        # Convert the solid mesh to meshpy objects. Without this parameter no
        # solid VTK file would be written.
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

        # Compare the vtk files.
        compare_vtk(self, 'test_meshpy_vtk_elements_solid', ref_file, vtk_file)

    def test_vtk_curve_cell_data(self):
        """
        Test that when creating a beam, cell data can be given.
        This test also checks, that the nan values in vtk can be explicitly
        given.
        """

        # Create the mesh.
        mesh = Mesh()
        mpy.vtk_nan_float = 69.69
        mpy.vtk_nan_int = 69

        # Add content to the mesh.
        mat = MaterialBeam(radius=0.05)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0], [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 1, 0], [2, 1, 0], n_el=2,
            vtk_cell_data={'cell_data': (1, mpy.vtk_type.int)})
        create_beam_mesh_arc_segment(mesh, Beam3rHerm2Line3, mat,
            [0, 2, 0], Rotation([1, 0, 0], np.pi), 1.5, np.pi / 2.0, n_el=2,
            vtk_cell_data={'cell_data': (2, mpy.vtk_type.int),
                'other_data': 69})

        # Write VTK output, with coupling sets."""
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_curve_cell_data_reference.vtu')
        vtk_file = os.path.join(testing_temp,
            'test_meshpy_vtk_curve_cell_data_beam.vtu')
        mesh.write_vtk(output_name='test_meshpy_vtk_curve_cell_data',
            output_directory=testing_temp, ascii=True)

        # Compare the vtk files.
        compare_vtk(self, 'test_meshpy_vtk_curve_cell_data', ref_file,
            vtk_file)

    def test_cubitpy_import(self):
        """
        Check that a import from a cubitpy object is the same as importing the
        dat file.
        """

        # Check if cubitpy can be loaded.
        import importlib
        found = importlib.util.find_spec('cubitpy') is not None
        if not found:
            # In this case skip the test.
            self.skipTest('CubitPy could not be loaded!')

        # Load the mesh creation functions.
        from tests.create_cubit_input import create_tube, create_tube_cubit

        # Create the input file and read the file.
        file_path = os.path.join(testing_temp, 'test_cubitpy_import.dat')
        create_tube(file_path)
        input_file = InputFile(dat_file=file_path)

        # Create the input file and read the cubit object.
        input_file_cubit = InputFile(cubit=create_tube_cubit())

        # Load the file from the reference folder.
        file_path_ref = os.path.join(testing_input,
            'baci_input_solid_tube.dat')
        input_file_ref = InputFile(dat_file=file_path_ref)

        # Compare the input files.
        compare_strings(self, 'test_cubitpy_import',
            input_file.get_string(header=False),
            input_file_cubit.get_string(header=False))
        compare_strings(self, 'test_cubitpy_import_reference',
            input_file.get_string(header=False),
            input_file_ref.get_string(header=False))

    def test_deep_copy(self):
        """
        Thist test checks that the deep copy function on a mesh does not copy
        the materials or functions.
        """

        # Create material and function object.
        mat = MaterialReissner(youngs_modulus=1, radius=1)
        fun = Function('COMPONENT 0 FUNCTION t')

        def create_mesh(mesh):
            """Add material and function to the mesh and create a beam."""
            mesh.add(fun, mat)
            set1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
                [0, 0, 0], [1, 0, 0])
            set2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
                [1, 0, 0], [1, 1, 0])
            mesh.add(BoundaryCondition(set1['line'], 'fix',
                bc_type=mpy.bc.dirichlet))
            mesh.add(BoundaryCondition(set2['line'], 'load',
                bc_type=mpy.bc.neumann))
            mesh.couple_nodes()

        # The second mesh will be translated and rotated with those vales.
        translate = [1., 2.34535435, 3.345353]
        rotation = Rotation([1, 0.2342342423, -2.234234], np.pi / 15 * 27)

        # First create the mesh twice, move one and get the input file.
        mesh_ref_1 = Mesh()
        mesh_ref_2 = Mesh()
        create_mesh(mesh_ref_1)
        create_mesh(mesh_ref_2)
        mesh_ref_2.rotate(rotation)
        mesh_ref_2.translate(translate)
        input_file_ref = InputFile()
        input_file_ref.add(mesh_ref_1, mesh_ref_2)

        # Now copy the first mesh and add them together in the input file.
        mesh_copy_1 = Mesh()
        create_mesh(mesh_copy_1)
        mesh_copy_2 = mesh_copy_1.copy()
        mesh_copy_2.rotate(rotation)
        mesh_copy_2.translate(translate)
        input_file_copy = InputFile()
        input_file_copy.add(mesh_copy_1, mesh_copy_2)

        # Check that the input files are the same.
        compare_strings(self,
            'test_deep_copy',
            input_file_ref.get_string(header=False, dat_header=False,
                add_script_to_header=False),
            input_file_copy.get_string(header=False, dat_header=False,
                add_script_to_header=False)
            )

    def test_mesh_add_checks(self):
        """
        This test checks that Mesh raises an error when double objects are
        added to the mesh.
        """

        # Mesh instance for this test.
        mesh = Mesh()

        # Create objects that will be added to the mesh.
        node = Node([0, 1., 2.])
        element = Beam()
        coupling = Coupling(mesh.nodes, mpy.coupling.fix)
        geometry_set = GeometrySet(mpy.geo.point)

        # Add each object once.
        mesh.add(node)
        mesh.add(element)
        mesh.add(coupling)
        mesh.add(geometry_set)

        # Add the objects again and check for errors.
        self.assertRaises(ValueError, mesh.add, node)
        self.assertRaises(ValueError, mesh.add, element)
        self.assertRaises(ValueError, mesh.add, coupling)
        self.assertRaises(ValueError, mesh.add, geometry_set)

    def test_check_two_couplings(self):
        """
        The current implementation can not handle more than one coupling on a
        node correctly, therefore we need to throw an error.
        """

        # Create mesh object.
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add two beams to create an elbow structure. The beams each have a
        # node at the intersection.
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [1, 0, 0], [1, 1, 0])

        # Call coupling twice -> this will create two coupling objects for the
        # corner node.
        mesh.couple_nodes()
        mesh.couple_nodes()

        # Create the input file. This will cause an error, as there are two
        # couplings for one node.
        self.assertRaises(ValueError, mesh.write_input_file, '/tmp/temp.dat',
                    add_script_to_header=False)

    def test_check_double_elements(self):
        """
        Check if there are overlapping elements in a mesh.
        """

        # Create mesh object.
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add two beams to create an elbow structure. The beams each have a
        # node at the intersection.
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0], [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0], [1, 0, 0])

        # Rotate the mesh with an arbitrary rotation.
        mesh.rotate(Rotation([1, 2, 3.24313], 2.2323423), [1, 3, -2.23232323])

        # The elements in the created mesh are overlapping, check that an error
        # is thrown.
        self.assertRaises(ValueError, mesh.check_overlapping_elements)

        # Check if the overlapping elements are written to the vtk output.
        warnings.filterwarnings("ignore")
        ref_file = os.path.join(testing_input,
            'test_meshpy_vtk_element_overlap_reference.vtu')
        vtk_file = os.path.join(testing_temp,
            'test_meshpy_vtk_element_overlap_beam.vtu')
        mesh.write_vtk(output_name='test_meshpy_vtk_element_overlap',
            output_directory=testing_temp, ascii=True,
            overlapping_elements=True)

        # Compare the vtk files.
        compare_vtk(self, 'test_check_double_elements', ref_file, vtk_file)

    def perform_test_check_overlapping_coupling_nodes(self, check=True):
        """
        Per default, we check that coupling nodes are at the same physical
        position. This check can be deactivated with the keyword
        check_overlapping_nodes when creating a Coupling.
        """

        # Create mesh object.
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add two beams to create an elbow structure. The beams each have a
        # node at the intersection.
        set_1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [0, 0, 0], [1, 0, 0])
        set_2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat,
            [2, 0, 0], [3, 0, 0])

        # Couple two nodes that are not at the same position.

        # Create the input file. This will cause an error, as there are two
        # couplings for one node.
        args = [
            [set_1['start'].nodes[0], set_2['end'].nodes[0]],
            'coupling_type_string'
            ]
        if check:
            self.assertRaises(ValueError, Coupling, *args)
        else:
            Coupling(*args, check_overlapping_nodes=False)

    def test_check_overlapping_coupling_nodes(self):
        """
        Perform the test that the coupling nodes can be tested if they are at
        the same position.
        """
        self.perform_test_check_overlapping_coupling_nodes(True)
        self.perform_test_check_overlapping_coupling_nodes(False)


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
