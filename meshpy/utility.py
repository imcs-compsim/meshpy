# -*- coding: utf-8 -*-
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
from .meshpy.find_close_nodes import find_close_nodes, find_close_nodes_binning


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


def get_close_nodes(nodes, binning=mpy.binning, nx=mpy.binning_n_bin,
        ny=mpy.binning_n_bin, nz=mpy.binning_n_bin, eps=mpy.eps_pos,
        return_nodes=True):
    """
    Find nodes that are close to each other.

    Args
    ----
    nodes: list(Node), np.array
        Nodes that are checked for partners, or numpy array of coordinates.
    binning: bool
        If binning should be used.
    nx, ny, nz: int
        Number of bins in the directions.
    eps: double
        Spherical value that the nodes have to be within, to be identified
        as overlapping.
    return_nodes: bool
        If true, the Node objects are returned, otherwise the index of the node
        objects in the list nodes. This option is mainly used for testing.

    Return
    ----
    partner_nodes: list(list(Node)), list(int)
        A list of lists with partner nodes.
    """

    if isinstance(nodes, np.ndarray):
        # Array is already given.
        coords = nodes
    else:
        # Get coordinates form nodes.
        coords = np.zeros([len(nodes), 3])
        for i, node in enumerate(nodes):
            coords[i, :] = node.coordinates

    if len(nodes) > mpy.binning_max_nodes_brute_force and not mpy.binning:
        warnings.warn('The function get_close_nodes is called directly '
            + 'with {} nodes, for performance reasons the '.format(len(nodes))
            + 'function find_close_nodes_binning should be used!')

    # Get list of closest pairs.
    if binning:
        has_partner, n_partner = find_close_nodes_binning(coords, nx, ny, nz,
            eps)
    else:
        has_partner, n_partner = find_close_nodes(coords, eps=eps)

    if return_nodes:
        # This is only possible if a list of nodes was given.
        if not isinstance(nodes, list):
            raise ValueError('The partner nodes can only be returned if a '
                + 'list of nodes was given as input!')

        # Create list with nodes.
        partner_nodes = [[] for i in range(n_partner)]
        for i, node in enumerate(nodes):
            if not has_partner[i] == -1:
                partner_nodes[has_partner[i]].append(node)

        return partner_nodes
    else:
        # Return the partner list.
        return has_partner, n_partner
