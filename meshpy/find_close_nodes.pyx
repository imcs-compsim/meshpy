# -*- coding: utf-8 -*-
"""
This script contains functions that calculate points that are close together. It
uses cython to gain performance.
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




# 
# @cython.boundscheck(False)  # Deactivate bounds checking.
# @cython.wraparound(False)   # Deactivate negative indexing.
# def find_close_nodes_segment(np.ndarray[FLOAT_t, ndim=2] coords, sections=20,FLOAT_t eps=1e-10):
#     
#     # Define types of variables for this function.
#     cdef int n_nodes, i, j
#     cdef int[3] n_sections, segment_index
#     cdef double[3][2] min_max_coord
#     cdef double[3] length
#     cdef double grid_offset
#     
#     # Offset for grid, so that even numbers wont be hit as much.
#     grid_offset = 89 * eps
#     
#     # Number of nodes.
#     n_nodes = len(coords) 
#     
#     # Set sections from input.
#     if isinstance(sections, int):
#         for i in range(3):
#             n_sections[i] = sections
#     elif len(sections) == 3:
#         for i in range(3):
#             n_sections[i] = sections[i]
#     else:
#         raise ValueError('Sections is wrong!')
#     
#     # Get min and max values of coordinates.
#     for i in range(n_nodes):
#         for j in range(3):
#             # First set the values to the first node
#             if i == 0:
#                 min_max_coord[j][0] = coords[i,j]
#                 min_max_coord[j][1] = coords[i,j]
#             else:
#                 if min_max_coord[j][0] > coords[i,j]:
#                     min_max_coord[j][0] = coords[i,j]
#                 elif min_max_coord[j][1] < coords[i,j]:
#                     min_max_coord[j][1] = coords[i,j]
#     
#     # Check if number of sections should be changed and define lengths
#     for i in range(3):
#         if abs(min_max_coord[i][1] - min_max_coord[i][0]) < 10*eps:
#             n_sections[i] = 1
#         length[i] = (min_max_coord[i][1] - min_max_coord[i][0]) / n_sections[i]
#     
#     # Create array and fill up with coordinates.
#     cdef np.ndarray[INT_t, ndim=3] segment_number_of_nodes = np.zeros(
#         [n_sections[0], n_sections[1], n_sections[2]], dtype=INT)
#     for i in range(n_nodes):
#         for j in range(3):
#             segment_index[j] = min(
#                 int(floor(coords[i,j] - min_max_coord[j][0] + 10*eps)),
#                 n_sections[j] - 1
#                 )
#             segment_nodes[i] += n_sections[j] * segment_index[j]
#         #print(segment_nodes)
#         #segment_number_of_nodes[segment_index[0], segment_index[1],
#             #segment_index[2]] += 1
#     print(segment_nodes)
#     




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
            for j in range(i+1, n_nodes):
                # Calculate the distance between the two nodes.
                distance = 0.
                for k in range(3):
                    distance += (coords[i,k] - coords[j,k])**2
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





















# 
# def find_close_nodes(np.ndarray[FLOAT_t, ndim=2] coords, FLOAT_t eps=1e-10,
#         INT_t n_sections=10):
#     
#     # Get min and max coordinate values
#     cdef np.ndarray[INT_t, n_dim=2] min_max = [ np.min(coords[:,i]), np.max(coords[:,i]) ]
#     
#     print(min_max)
#     print(eps)
#     # Number of nodes.
#     cdef int n_nodes, i, j
#     cdef FLOAT_t value
#     n_nodes = len(coords)
#     value = 0.
#     for i in range(n_nodes):
#         for j in range(i, n_nodes):
#             value = coords[j, 0] + coords[j, 2] + coords[j, 2]
#     
#     # Cython access to input array.
#     #cdef double [:, :] narr_view = coords
#     
#     
#     #narr_view[1,1] = 2.
#     
#     print(value)
#     print(n_nodes)
#     
#     
# 


