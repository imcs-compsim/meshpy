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
from .find_close_points_cython_lib import (
    find_close_points as find_close_points_cython,
    find_close_points_binning as find_close_points_binning_cython,
)


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
    nodes,
    binning=mpy.binning,
    nx=mpy.binning_n_bin,
    ny=mpy.binning_n_bin,
    nz=mpy.binning_n_bin,
    eps=mpy.eps_pos,
):
    """
    Find n-dimensional points that are close to each other.

    Args
    ----
    nodes: np.array(n_points x n_dim)
        Point coordinates that are checked for partners.
    binning: bool
        If binning should be used. Only the first three dimensions will be used
        for binning.
    nx, ny, nz: int
        Number of bins in the first three dimensions.
    eps: double
        Hypersphere radius value that the coordinates have to be within, to be
        identified as overlapping.

    Return
    ----
    partner_indices: list(list(int))
        A list of lists of point indices that are close to each other.
    """

    if len(nodes) > mpy.binning_max_nodes_brute_force and not mpy.binning:
        warnings.warn(
            "The function get_close_points is called directly "
            + "with {} points, for performance reasons the ".format(len(nodes))
            + "function find_close_points_binning should be used!"
        )

    # Get list of closest pairs.
    if binning:
        has_partner, n_partner = find_close_points_binning_cython(
            nodes, nx, ny, nz, eps
        )
    else:
        has_partner, n_partner = find_close_points_cython(nodes, eps=eps)

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
