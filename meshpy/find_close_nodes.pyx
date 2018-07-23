# -*- coding: utf-8 -*-
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
INT = np.int
ctypedef np.int_t INT_t


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
def find_close_nodes(np.ndarray[FLOAT_t, ndim=2] coords, FLOAT_t eps=1e-10):
    """
    Finds coordinates that are within an tolerance of each other.

    Args
    ----
    coords: numpy array
        Array with the coordinates of the nodes.
    eps: float
        Tolerance to look for neighbors.

    Return
    ----
    partners: numpy array
        An array with integers, marking the set a node is part of. -1 means the
        node does not have a partner.
    """

    # Define types of variables for this function.
    cdef int n_nodes, partner, this_is_partner, i, j, k
    cdef double distance

    # Number of nodes.
    n_nodes = len(coords)

    # This vector is 0 if a node does not belong to a pair, otherwise it is the
    # number of the pair.
    cdef np.ndarray[INT_t, ndim=1] has_partner = (
        np.zeros(n_nodes, dtype=INT) - 1)

    # Loop over nodes
    partner = 0
    for i in range(n_nodes):
        this_is_partner = 0
        if has_partner[i] == -1:
            for j in range(i + 1, n_nodes):
                # Calculate the distance between the two nodes.
                distance = 0.
                for k in range(3):
                    distance += (coords[i, k] - coords[j, k])**2
                distance = sqrt(distance)
                # Check if the distance is smaller than the threshold, and add
                # to has_partner list.
                if distance < eps:
                    this_is_partner = 1
                    has_partner[j] = partner
            # If this one has a partner set this node too.
            if this_is_partner == 1:
                has_partner[i] = partner
                partner += 1

    return has_partner, partner
