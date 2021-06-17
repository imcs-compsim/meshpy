# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This module implements some basic functions that are used in the meshpy
application.
"""

# Python modules.
import subprocess
import os
import numpy as np
import warnings

# Meshpy modules.
from .conf import mpy
from .meshpy import find_close_points as find_points


def get_git_data(repo):
    """Return the hash and date of the current git commit."""
    out_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out_date = subprocess.run(['git', 'show', '-s', '--format=%ci'], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if not out_sha.returncode + out_date.returncode == 0:
        return None, None
    else:
        sha = out_sha.stdout.decode('ascii').strip()
        date = out_date.stdout.decode('ascii').strip()
        return sha, date


# Set the git version in the global configuration object.
mpy.git_sha, mpy.git_date = get_git_data(
    os.path.dirname(os.path.realpath(__file__)))


def flatten(data):
    """Flatten out all list items in data."""
    flatten_list = []
    if type(data) == list:
        for item in data:
            flatten_list.extend(flatten(item))
        return flatten_list
    else:
        return [data]


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


def find_close_points(nodes, binning=mpy.binning, nx=mpy.binning_n_bin,
        ny=mpy.binning_n_bin, nz=mpy.binning_n_bin, eps=mpy.eps_pos):
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
        warnings.warn('The function get_close_points is called directly '
            + 'with {} points, for performance reasons the '.format(len(nodes))
            + 'function find_close_points_binning should be used!')

    # Get list of closest pairs.
    if binning:
        has_partner, n_partner = find_points.find_close_points_binning(nodes,
            nx, ny, nz, eps)
    else:
        has_partner, n_partner = find_points.find_close_points(nodes, eps=eps)

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


def check_node_by_coordinate(node, axis, value, eps=1e-10):
    """
    Check if the node is at a certain coordinate value.

    Args
    ----
    node: Node
        The node to be checked for its position.
    axis: int
        Coordinate axis to check.
        0 -> x, 1 -> y, 2 -> z
    value: float
        Value for the coordinate that the node should have.
    eps: float
        Tolerance to check for equality.
    """
    if np.abs(node.coordinates[axis] - value) < eps:
        return True
    else:
        return False


def get_min_max_coordinates(nodes):
    """
    Return an array with the minimal and maximal coordinates of the given
    nodes.

    Return
    ----
    min_max_coordinates:
        [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    coordinates = np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coordinates[i, :] = node.coordinates
    min_max = np.zeros(6)
    min_max[:3] = np.min(coordinates, axis=0)
    min_max[3:] = np.max(coordinates, axis=0)
    return min_max
