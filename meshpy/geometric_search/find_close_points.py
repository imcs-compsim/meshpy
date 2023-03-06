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
Find points in a point cloud that are at the same position (within a certain tolerance).
"""

# Python modules
import numpy as np
import warnings

# Meshpy modules
from ..conf import mpy

# Geometric search modules
# Cython modules
from .geometric_search_cython import cython_available

if cython_available:
    from .geometric_search_cython import (
        find_close_points as find_close_points_cython,
        find_close_points_binning as find_close_points_binning_cython,
    )

# ArborX
from .geometric_search_arborx import arborx_available

if arborx_available:
    from .geometric_search_arborx import find_close_points_arborx


def point_partners_to_partner_indices(point_partners, n_partners):
    """
    Convert the partner indices for each point to a list of lists with the
    indices for all partners.
    """
    partner_indices = [[] for i in range(n_partners)]
    for i, partner_index in enumerate(point_partners):
        if partner_index != -1:
            partner_indices[partner_index].append(i)
    return partner_indices


def partner_indices_to_point_partners(partner_indices, n_points):
    """
    Convert the list of lists with the indices for all partners to the partner
    indices for each point.
    """
    point_partners = [-1 for _i in range(n_points)]
    for i_partner, partners in enumerate(partner_indices):
        for index in partners:
            point_partners[index] = i_partner
    return point_partners, len(partner_indices)


def find_close_points(
    point_coordinates,
    algorithm=mpy.geometric_search_algorithm.automatic,
    n_bins=[mpy.geometric_search_binning_n_bin] * 3,
    tol=mpy.eps_pos,
):
    """
    Find n-dimensional points that are close to each other.

    Args
    ----
    point_coordinates: np.array(n_points x n_dim)
        Point coordinates that are checked for partners.
    algorithm: mpy.GeometricSearchAlgorithm
        Type of geometric search algorithm that should be used.
    n_bins: list(int)
        Number of bins in the first three dimensions.
    tol: double
        Hypersphere radius value that the coordinates have to be within, to be
        identified as overlapping. Be careful when using an arborx search
        algorithm, das the tolerance there is of type float, not double.

    Return
    ----
    partner_indices: list(list(int))
        A list of lists of point indices that are close to each other.
    """

    n_points = len(point_coordinates)
    n_dim = point_coordinates.shape[1]

    if algorithm is mpy.geometric_search_algorithm.automatic:
        # Decide which algorithm to use
        if n_points < 100:
            algorithm = mpy.geometric_search_algorithm.brute_force_cython
        elif arborx_available and n_dim == 3:
            algorithm = mpy.geometric_search_algorithm.boundary_volume_hierarchy_arborx
        else:
            algorithm = mpy.geometric_search_algorithm.binning_cython

    if (
        algorithm is mpy.geometric_search_algorithm.brute_force_cython
        and n_points > mpy.geometric_search_max_nodes_brute_force
    ):
        warnings.warn(
            "The function find_close_points is called with the brute force algorithm "
            + f"with {n_points} points, for performance reasons other algorithms should be used!"
        )

    if (
        algorithm is mpy.geometric_search_algorithm.boundary_volume_hierarchy_arborx
        and not n_dim == 3
    ):
        raise ValueError(
            "ArborX geometric search is currently only implemented for 3 dimensions"
        )
    elif (
        algorithm is mpy.geometric_search_algorithm.boundary_volume_hierarchy_arborx
        and not arborx_available
    ):
        raise ValueError("ArborX geometric search is not available")
    elif (
        algorithm is mpy.geometric_search_algorithm.brute_force_cython
        or algorithm is mpy.geometric_search_algorithm.brute_force_cython
    ) and not cython_available:
        raise ValueError("Cython geometric search is not available")

    # Get list of closest pairs
    if algorithm is mpy.geometric_search_algorithm.brute_force_cython:
        has_partner, n_partner = find_close_points_cython(point_coordinates, tol)
    elif algorithm is mpy.geometric_search_algorithm.binning_cython:
        has_partner, n_partner = find_close_points_binning_cython(
            point_coordinates, *n_bins, tol
        )
    elif algorithm is mpy.geometric_search_algorithm.boundary_volume_hierarchy_arborx:
        has_partner, n_partner = find_close_points_arborx(point_coordinates, tol)
    else:
        raise TypeError("Got unexpected algorithm")

    return point_partners_to_partner_indices(has_partner, n_partner)


def find_close_nodes(nodes, **kwargs):
    """
    Find nodes that are close to each other.

    Args
    ----
    nodes: list(Node)
        Nodes that are checked for partners.
    **kwargs:
        Arguments passed on to find_close_points

    Return
    ----
    partner_nodes: list(list(Node))
        A list of lists of nodes that are close to each other.
    """

    coords = np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coords[i, :] = node.coordinates
    partner_indices = find_close_points(coords, **kwargs)
    return [[nodes[i] for i in partners] for partners in partner_indices]
