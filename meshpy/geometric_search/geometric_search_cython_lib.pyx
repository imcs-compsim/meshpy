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
# cython: language_level=3
"""
This script contains functions that calculate points that are close together.
It uses cython to gain performance.
"""


# Python imports.
import numpy as np
from libc.math cimport sqrt, abs, floor

# Cython imports.
cimport cython
cimport numpy as np


# Define float type
FLOAT = np.float64
ctypedef np.float64_t FLOAT_t

# Define integer type
INT = np.int64
ctypedef np.int_t INT_t


@cython.wraparound(False)   # Deactivate negative indexing.
def find_close_points(np.ndarray[FLOAT_t, ndim=2] coords, FLOAT_t eps):
    """
    Finds coordinates that are within a tolerance of each other.

    Args
    ----
    coords: numpy array
        Array with the coordinates of the nodes.
    eps: float
        Tolerance to look for neighbors.

    Return
    ----
    has_partner: numpy array
        An array with integers, marking the set a node is part of. -1 means the
        node does not have a partner.
    partner: int
        Number of clusters found.
    """

    # Define types of variables for this function.
    cdef int n_nodes, partner, this_is_partner, i, j, k, n_dim
    cdef double distance

    # Number of nodes and dimension of coordinates.
    n_nodes = np.shape(coords)[0]
    n_dim = np.shape(coords)[1]

    # This vector is -1 if a node does not belong to a pair, otherwise it is
    # the number of the pair.
    cdef np.ndarray[INT_t, ndim=1] has_partner = (
        np.zeros(n_nodes, dtype=INT) - 1)

    # Loop over nodes.
    partner = 0
    for i in range(n_nodes):
        this_is_partner = 0
        if has_partner[i] == -1:
            for j in range(i + 1, n_nodes):
                # Calculate the distance between the two nodes.
                distance = 0.
                for k in range(n_dim):
                    distance += (coords[i, k] - coords[j, k])**2
                distance = sqrt(distance)
                # Check if the distance is smaller than the threshold, and add
                # to has_partner list.
                if distance < eps:
                    this_is_partner = 1
                    if not has_partner[j] == -1:
                        raise RuntimeError('The case where a node connects two'
                            ' other nodes that are more than eps apart is not'
                            ' yet implemented. Check if the value for eps'
                            ' makes physical sense!')
                    has_partner[j] = partner
            # If this one has a partner set this node too.
            if this_is_partner == 1:
                has_partner[i] = partner
                partner += 1

    return has_partner, partner


@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Modulo
def find_close_points_binning(np.ndarray[FLOAT_t, ndim=2] coords,
        FLOAT_t eps, INT_t nx, INT_t ny, INT_t nz):
    """
    Finds coordinates that are within an tolerance of each other. Create
    nx * ny * nz bins and look for neighbors in the bins. This speeds up the
    execution considerably. Even if the coordinates have more than 3 columns,
    the bins will only be created for the first 3.

    Args
    ----
    coords: numpy array
        Array with the coordinates of the nodes.
    eps: float
        Tolerance to look for neighbors.
    nx, ny, nz: int
        Number of bins in the directions.

    Return
    ----
    has_partner: numpy array
        An array with integers, marking the set a node is part of. -1 means the
        node does not have a partner.
    partner: int
        Number of pairs found.
    """

    # Define types of variables for this function.
    cdef int i, j, k, counter, n_nodes, n_bin, node_index, partner, \
        partner_bin, partner_at_start_bin, n_nodes_in_this_bin, \
        global_partner_id, local_partner_index, partner_counter, n_dim

    # Define arrays for this function.
    cdef np.ndarray[INT_t, ndim=1] n_bin_xyz, node_index_xyz, \
        nodes_in_this_bin, has_partner, has_partner_bin, \
        local_to_global_partner_id, partner_counter_list
    cdef np.ndarray[INT_t, ndim=2] nodes_bin, fac_xyz
    cdef np.ndarray[FLOAT_t, ndim=1] max_coord, min_coord, h_bin
    cdef np.ndarray[FLOAT_t, ndim=2] find_close_points_coords

    # Set array with number of bins.
    n_bin_xyz = np.zeros(3, dtype=INT)
    n_bin_xyz[0] = nx
    n_bin_xyz[1] = ny
    n_bin_xyz[2] = nz

    # Get max and min coordinates of points.
    max_coord = np.max(coords[:, :3], 0)
    min_coord = np.min(coords[:, :3], 0)

    # Get size of bins.
    h_bin = (max_coord - min_coord) / n_bin_xyz

    # Check that the bin size is more than 10 * eps.
    for i in range(3):
        if h_bin[i] < 10 * eps:
            h_bin[i] = max_coord[i] - min_coord[i]
            n_bin_xyz[i] = 1
    n_bin = n_bin_xyz[0] * n_bin_xyz[1] * n_bin_xyz[2]

    # Set the permutated factor array.
    counter = 0
    fac_xyz = np.zeros([8, 3], dtype=INT)
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                fac_xyz[counter, :] = [i, j, k]
                counter += 1

    # Check in which bin a node belongs.
    n_nodes = len(coords)
    n_dim = np.shape(coords)[1]
    nodes_bin = np.zeros([n_nodes, 8], dtype=INT) - 1
    node_index_xyz = np.zeros(3, dtype=INT)
    nodes_in_bin = [[] for i in range(n_bin)]
    for i in range(n_nodes):

        # Loop over cases for boundary.
        for j in range(8):
            for k in range(3):
                # This array stores the index for the bin in x, y and z
                # direction.
                node_index_xyz[k] = int(floor(
                    (coords[i, k] - min_coord[k] + fac_xyz[j, k] * eps)
                    / h_bin[k])
                    )
                if node_index_xyz[k] > n_bin_xyz[k] - 1:
                    node_index_xyz[k] = n_bin_xyz[k] - 1
                elif node_index_xyz[k] < 0:
                    node_index_xyz[k] = 0

            # Calculate the global bin index.
            node_index = (
                node_index_xyz[0] * (n_bin_xyz[1] * n_bin_xyz[2])
                + node_index_xyz[1] * n_bin_xyz[2] + node_index_xyz[2]
                )

            # Add to the indices of this node.
            for k in range(8):
                if nodes_bin[i, k] == node_index:
                    break
                if nodes_bin[i, k] == -1:
                    nodes_bin[i, k] = node_index
                    nodes_in_bin[node_index].append(i)
                    break

    # Look for double nodes in bins:
    has_partner = (np.zeros(n_nodes, dtype=INT) - 1)
    partner = 0
    for i in range(n_bin):

        # Number of nodes to check in this bin.
        n_nodes_in_this_bin = len(nodes_in_bin[i])

        if n_nodes_in_this_bin > 0:

            # Get an array with the node number for this bin.
            nodes_in_this_bin = np.zeros(n_nodes_in_this_bin, dtype=INT)
            for j in range(n_nodes_in_this_bin):
                nodes_in_this_bin[j] = nodes_in_bin[i][j]

            # Get the coordinates for this bin.
            find_close_points_coords = np.zeros([n_nodes_in_this_bin, n_dim],
                dtype=FLOAT)
            for j in range(n_nodes_in_this_bin):
                for k in range(n_dim):
                    find_close_points_coords[j, k] = (
                        coords[nodes_in_bin[i][j], k]
                        )
            has_partner_bin, partner_bin = find_close_points(
                find_close_points_coords, eps)

            # Set empty array for local to global ids.
            local_to_global_partner_id = np.zeros(partner_bin, dtype=INT) - 1

            # Set index of found partners sets.
            partner_at_start_bin = partner

            # Check if one of the nodes already has a partner.
            for j in range(n_nodes_in_this_bin):

                # Index of the local partner.
                local_partner_index = has_partner_bin[j]

                if not local_partner_index == -1:

                    # Index of the global partner.
                    global_partner_id = has_partner[nodes_in_this_bin[j]]

                    if not global_partner_id == -1:
                        # A global partner exists for this node. Check the
                        # local to global value.
                        if global_partner_id == local_to_global_partner_id[
                                local_partner_index]:
                            # Ok, the partners have already been found.
                            pass
                        elif local_to_global_partner_id[
                                local_partner_index] == -1:
                            # No global partner has been found yet, insert
                            # the global number.
                            local_to_global_partner_id[
                                local_partner_index] = global_partner_id
                        elif (local_to_global_partner_id[local_partner_index]
                                >= partner_at_start_bin):
                            # Partner were only found in this bin.
                            pass
                        else:
                            # This node connects two different partner sets,
                            # this is not implemented,
                            raise NotImplementedError('Not implemented!')
                    elif local_to_global_partner_id[local_partner_index] == -1:
                        # No global partner exists for this code.
                        # No local to global indices set yet, add new one.
                        local_to_global_partner_id[
                            local_partner_index] = partner
                        partner += 1

            # Write local to global parters.
            for j in range(n_nodes_in_this_bin):

                # Index of the local partner.
                local_partner_index = has_partner_bin[j]

                if not local_partner_index == -1:

                    has_partner[nodes_in_this_bin[j]] = \
                        local_to_global_partner_id[local_partner_index]

    # Renumber the partners so results can be compared with
    # find_close_points().
    partner_counter_list = np.zeros(partner, dtype=INT) - 1
    partner_counter = 0
    for i in range(n_nodes):
        if not has_partner[i] == -1:
            # Check if there is already a renumbering.
            if partner_counter_list[has_partner[i]] == -1:
                # No renumbering available, set to counter.
                partner_counter_list[has_partner[i]] = partner_counter
                has_partner[i] = partner_counter
                partner_counter += 1
            else:
                has_partner[i] = partner_counter_list[has_partner[i]]

    return has_partner, partner
