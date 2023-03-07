# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
This script is used to test the functionality of the meshpy.geometric_search module.
"""

# Python imports
import unittest
import numpy as np
import random

# Meshpy imports
from meshpy import mpy, Rotation, Mesh, MaterialReissner, Beam3rHerm2Line3
from meshpy.mesh_creation_functions.beam_honeycomb import (
    create_beam_mesh_honeycomb_flat,
)
from meshpy.geometric_search import (
    partner_indices_to_point_partners,
    point_partners_to_partner_indices,
    find_close_points,
    cython_available,
    arborx_available,
)


# Testing imports
# from testing_utility import (
#     skip_fail_test,
#     testing_temp,
#     testing_input,
#     compare_strings,
#     compare_vtk,
# )


class TestGeometricSearch(unittest.TestCase):
    """Test various stuff from the meshpy.geometric_search module."""

    def setUp(self):
        """
        This method is called before each test and sets the default meshpy
        values for each test. The values can be changed in the individual
        tests.
        """

        # Set default values for global parameters.
        mpy.set_default_values()
        mpy.geometric_search_max_nodes_brute_force = 1010

    def test_find_close_points_between_bins_brute_force_cython(self):
        if cython_available:
            self.xtest_find_close_points_between_bins(
                mpy.geometric_search_algorithm.brute_force_cython
            )
        else:
            self.skipTest("Cython not available")

    def test_find_close_points_between_bins_binning_cython(self):
        if cython_available:
            self.xtest_find_close_points_between_bins(
                mpy.geometric_search_algorithm.binning_cython, n_bins=[4, 4, 4]
            )
        else:
            self.skipTest("Cython not available")

    def test_find_close_points_between_bins_boundary_volume_hierarchy_arborx(self):
        if arborx_available:
            self.xtest_find_close_points_between_bins(
                mpy.geometric_search_algorithm.boundary_volume_hierarchy_arborx
            )
        else:
            self.skipTest("ArborX not available")

    def xtest_find_close_points_between_bins(self, algorithm, **kwargs):
        """
        Test if the find_close_points function returns the expected results.
        The points are chosen such that for n_bins = [4, 4, 4], some points
        are exactly at the boundary between bins. This test case can be used
        for all algorithms, not just binning.
        """

        # Set the seed for the pseudo random numbers
        random.seed(0)

        # Add random nodes to a cube with width 2. Randomly add nodes close to
        # existing nodes. The distance has to be shorter than 0.5 * eps_medium
        # since the algorithm needs all close nodes to be within a sphere with
        # radius eps_medium.
        eps_medium = 1e-5
        eps_medium_factor = 0.49 * eps_medium
        n_nodes_random = 999
        n_nodes_total = n_nodes_random + 2
        coords = np.zeros([n_nodes_total, 3])
        for i in range(n_nodes_random):
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
        diff = np.random.rand(n_nodes_random, 3)
        diff /= np.linalg.norm(diff, axis=1)[:, None]
        coords[:n_nodes_random] += eps_medium_factor * diff

        # Add nodes such that the bin will be exactly [-1, 1] x [-1, 1] x [-1, 1]
        coords[-1, :] = [1.0, 1.0, 1.0]
        coords[-2, :] = [-1.0, -1.0, -1.0]

        # Add nodes between the bins
        coords[-3, :] = [0.0, 0.0, 0.0]
        coords[-4, :] = [0.0, 0.0, 0.0]
        coords[-5, :] = [0.0, 0.0, 0.5]
        coords[-6, :] = [0.0, 0.0, 0.5]
        coords[-7, :] = [0.0, 0.5, 0.0]
        coords[-8, :] = [0.0, 0.5, 0.0]
        coords[-9, :] = [0.5, 0.0, 0.0]
        coords[-10, :] = [0.5, 0.0, 0.0]
        coords[-11, :] = [0.0 + eps_medium_factor, 0.0, 0.0]
        coords[-12, :] = [0.0, 0.0 + eps_medium_factor, 0.0]
        coords[-13, :] = [0.0, 0.0, 0.0 + eps_medium_factor]
        coords[-14, :] = [0.0 - eps_medium_factor, 0.0, 0.0]
        coords[-15, :] = [0.0, 0.0 - eps_medium_factor, 0.0]
        coords[-16, :] = [0.0, 0.0, 0.0 - eps_medium_factor]

        # Test the number of partners and list of partners
        partner_indices = find_close_points(
            coords, algorithm=algorithm, tol=eps_medium, **kwargs
        )

        # Compare to the reference solution.
        point_partners_reference = -1 * np.ones([n_nodes_total], dtype=int)
        # fmt: off
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
        # fmt: on
        point_partners_reference[index_vector] = val_vector
        point_partners, _n_partners = partner_indices_to_point_partners(
            partner_indices, len(coords)
        )
        self.assertEqual(len(partner_indices), 146)
        self.assertTrue(np.array_equal(point_partners, point_partners_reference))

    def test_find_close_points_binning_flat_brute_force_cython(self):
        if cython_available:
            self.xtest_find_close_points_binning_flat(
                mpy.geometric_search_algorithm.brute_force_cython
            )
        else:
            self.skipTest("Cython not available")

    def test_find_close_points_binning_flat_binning_cython(self):
        if cython_available:
            self.xtest_find_close_points_binning_flat(
                mpy.geometric_search_algorithm.binning_cython
            )
        else:
            self.skipTest("Cython not available")

    def test_find_close_points_binning_flat_boundary_volume_hierarchy_arborx(self):
        if arborx_available:
            self.xtest_find_close_points_binning_flat(
                mpy.geometric_search_algorithm.boundary_volume_hierarchy_arborx
            )
        else:
            self.skipTest("ArborX not available")

    def xtest_find_close_points_binning_flat(self, algorithm, **kwargs):
        """
        Test case for coupling of points, when the nodes are all on a plane. This is
        challenging for a binning based approach. However, this test case can also
        be used for all find_close_point algorithms.
        """

        # Dummy material.
        material = MaterialReissner(radius=0.1)

        def create_flat_mesh():
            """Create a flat honeycomb mesh."""
            mesh = Mesh()
            create_beam_mesh_honeycomb_flat(
                mesh, Beam3rHerm2Line3, material, 1, 5, 5, create_couplings=False
            )
            return mesh

        # Get a reference solution for the coupling nodes.
        # fmt: off
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
        # fmt: on
        reference_partners = point_partners_to_partner_indices(
            reference_partners_list, 66
        )

        partners = find_close_points(
            create_flat_mesh().get_global_coordinates(middle_nodes=False)[0],
            algorithm=algorithm,
            **kwargs
        )
        self.assertEqual(reference_partners, partners)

        # Apply different rotations and compare the partner results.
        rotations = [
            Rotation([1, 0, 0], 0.0),
            Rotation([1, 0, 0], np.pi * 0.5),
            Rotation([0, 1, 0], np.pi * 0.5),
            Rotation([0, 0, 1], np.pi * 0.5),
            Rotation([1, 3, -4], 25.21561 * np.pi * 0.5),
        ]
        for rotation in rotations:
            # Create and rotate the mesh
            mesh = create_flat_mesh()
            mesh.rotate(rotation)

            # The reference data was created for the nodes without the middle
            # nodes, therefore we filter the middle nodes here
            partners = find_close_points(
                mesh.get_global_coordinates(middle_nodes=False)[0],
                algorithm=algorithm,
                **kwargs
            )

            # Compare the partners with the reference.
            self.assertEqual(partners, reference_partners)

    def test_find_close_points_dimension_brute_force_cython(self):
        if cython_available:
            self.xtest_find_close_points_dimension(
                mpy.geometric_search_algorithm.brute_force_cython
            )
        else:
            self.skipTest("Cython not available")

    def test_find_close_points_dimension_binning_cython(self):
        if cython_available:
            self.xtest_find_close_points_dimension(
                mpy.geometric_search_algorithm.binning_cython
            )
        else:
            self.skipTest("Cython not available")

    def xtest_find_close_points_dimension(self, algorithm, **kwargs):
        """
        Test that the find_close_points function also works properly with
        multidimensional points.
        """

        # Set the seed for the pseudo random numbers.
        random.seed(0)

        # Create array with coordinates.
        coords = np.array(
            [
                [0.0, 0, 0, 0, 0, 0],
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
            ]
        )

        # Expected results
        # fmt: off
        has_partner_expected = [-1, -1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, 0,
            1, 2, 3, 4]
        # fmt: on
        partner_expected = 5

        # Get results
        partner_indices = find_close_points(coords, algorithm=algorithm, **kwargs)
        has_partner, partner = partner_indices_to_point_partners(
            partner_indices, len(coords)
        )

        # Check the results
        self.assertTrue(has_partner_expected, has_partner)
        self.assertEqual(partner_expected, partner)


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
