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
This module implements some basic functions that are used in the meshpy
application.
"""

# Python modules.
import subprocess
import os
import shutil
from pathlib import Path
import numpy as np

# Meshpy modules.
from .conf import mpy
from .node import Node, NodeCosserat
from .geometry_set import GeometrySet
from .geometric_search.find_close_points import (
    find_close_points,
    point_partners_to_partner_indices,
)


def get_git_data(repo):
    """Return the hash and date of the current git commit."""
    out_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    out_date = subprocess.run(
        ["git", "show", "-s", "--format=%ci"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if not out_sha.returncode + out_date.returncode == 0:
        return None, None
    else:
        sha = out_sha.stdout.decode("ascii").strip()
        date = out_date.stdout.decode("ascii").strip()
        return sha, date


# Set the git version in the global configuration object.
mpy.git_sha, mpy.git_date = get_git_data(os.path.dirname(os.path.realpath(__file__)))


def flatten(data):
    """Flatten out all list items in data."""
    flatten_list = []
    if type(data) == list:
        for item in data:
            flatten_list.extend(flatten(item))
        return flatten_list
    else:
        return [data]


def find_close_nodes(nodes, **kwargs):
    """
    Find nodes in a point cloud that are within a certain tolerance
    of each other.

    Args
    ----
    nodes: list(Node)
        Nodes who are part ot eh point cloud.
    **kwargs:
        Arguments passed on to geometric_search.find_close_points

    Return
    ----
    partner_nodes: list(list(Node))
        A list of lists of nodes that are close to each other, i.e.,
        each element in the returned list contains nodes that are close
        to each other.
    """

    coords = np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coords[i, :] = node.coordinates
    partner_indices = point_partners_to_partner_indices(
        *find_close_points(coords, **kwargs)
    )
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


def clean_simulation_directory(sim_dir):
    """
    If the simulation directory exists, the user is asked if the contents
    should be removed. If it does not exist, it is created.
    """

    # Check if simulation directory exists.
    if os.path.exists(sim_dir):
        print('Path "{}" already exists'.format(sim_dir))
        while True:
            answer = input("DELETE all contents? (y/n): ")
            if answer.lower() == "y":
                for filename in os.listdir(sim_dir):
                    file_path = os.path.join(sim_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        ValueError("Failed to delete %s. Reason: %s" % (file_path, e))
                return
            elif answer.lower() == "n":
                raise ValueError("Directory is not deleted!")
    else:
        Path(sim_dir).mkdir(parents=True, exist_ok=True)


def get_node(item, *, check_cosserat_node=False):
    """
    Function to get a node from the input variable. This function
    accepts a Node object as well as a GeometrySet object.

    Args
    ----
    item:
        This can be a GeometrySet with exactly one node or a single node object.
    check_cosserat: bool
        If a check should be performed, that the given node is a CosseratNode.
    """
    if isinstance(item, Node):
        node = item
    elif isinstance(item, GeometrySet):
        # Check if there is only one node in the set
        if len(item.nodes) == 1:
            node = item.nodes[0]
        else:
            raise ValueError("GeometrySet does not have exactly one node!")
    else:
        raise TypeError(
            'The given object can be node or GeometrySet got "{}"!'.format(type(item))
        )

    if check_cosserat_node and not isinstance(node, NodeCosserat):
        raise TypeError("Expected a NodeCosserat object.")

    return node
