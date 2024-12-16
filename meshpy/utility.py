# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
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
"""This module implements some basic functions that are used in the meshpy
application."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .conf import mpy
from .geometric_search.find_close_points import (
    find_close_points,
    point_partners_to_partner_indices,
)
from .geometry_set import GeometrySet, GeometrySetBase
from .node import Node, NodeCosserat


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


def find_close_nodes(nodes, **kwargs):
    """Find nodes in a point cloud that are within a certain tolerance of each
    other.

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


def check_node_by_coordinate(node, axis, value, eps=mpy.eps_pos):
    """Check if the node is at a certain coordinate value.

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
    return np.abs(node.coordinates[axis] - value) < eps


def get_min_max_coordinates(nodes):
    """Return an array with the minimal and maximal coordinates of the given
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


def clean_simulation_directory(sim_dir, *, ask_before_clean=False):
    """Clear the simulation directory. If it does not exist, it is created.
    Optionally the user can be asked before a deletion of files.

    Args
    ----
    sim_dir:
        Path to a directory
    ask_before_clean: bool
        Flag which indicates whether the user must confirm removal of files and directories
    """

    # Check if simulation directory exists.
    if os.path.exists(sim_dir):
        if ask_before_clean:
            print(f'Path "{sim_dir}" already exists')
        while True:
            if ask_before_clean:
                answer = input("DELETE all contents? (y/n): ")
            else:
                answer = "y"
            if answer.lower() == "y":
                for filename in os.listdir(sim_dir):
                    file_path = os.path.join(sim_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        raise ValueError(f"Failed to delete {file_path}. Reason: {e}")
                return
            elif answer.lower() == "n":
                raise ValueError("Directory is not deleted!")
    else:
        Path(sim_dir).mkdir(parents=True, exist_ok=True)


def get_single_node(item, *, check_cosserat_node=False):
    """Function to get a single node from the input variable. This function
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
    elif isinstance(item, GeometrySetBase):
        # Check if there is only one node in the set
        nodes = item.get_points()
        if len(nodes) == 1:
            node = nodes[0]
        else:
            raise ValueError("GeometrySet does not have exactly one node!")
    else:
        raise TypeError(
            f'The given object can be node or GeometrySet got "{type(item)}"!'
        )

    if check_cosserat_node and not isinstance(node, NodeCosserat):
        raise TypeError("Expected a NodeCosserat object.")

    return node


def filter_nodes(nodes, *, middle_nodes=True):
    """Filter the list of the given nodes. Be aware that if no filters are
    enabled the original list will be returned.

    Args
    ----
    nodes: list(Nodes)
        If this list is given it will be returned as is.
    middle_nodes: bool
        If middle nodes should be returned or not.
    """

    if not middle_nodes:
        return [node for node in nodes if middle_nodes or not node.is_middle_node]
    else:
        return nodes


def get_nodal_coordinates(nodes):
    """Return an array with the coordinates of the given nodes.

    Args
    ----
    kwargs:
        Will be passed to self.get_global_nodes.

    Return
    ----
    pos: np.array
        Numpy array with all the positions of the nodes.
    """
    coordinates = np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coordinates[i, :] = node.coordinates
    return coordinates


def get_nodal_quaternions(nodes):
    """Return an array with the quaternions of the given nodes.

    Args
    ----
    kwargs:
        Will be passed to self.get_global_nodes.

    Return
    ----
    pos: np.array
        Numpy array with all the positions of the nodes.
    """
    quaternions = np.zeros([len(nodes), 4])
    for i, node in enumerate(nodes):
        if isinstance(node, NodeCosserat):
            quaternions[i, :] = node.rotation.get_quaternion()
        else:
            # For the case of nodes that belong to solid elements,
            # we define the following default value:
            quaternions[i, :] = [2.0, 0.0, 0.0, 0.0]
    return quaternions


def get_nodes_by_function(nodes, function, *args, middle_nodes=False, **kwargs):
    """Return all nodes for which the function evaluates to true.

    Args
    ----
    nodes: [Node]
        Nodes that should be filtered.
    function: function(node, *args, **kwargs)
        Nodes for which this function is true are returned.
    middle_nodes: bool
        If this is true, middle nodes of a beam are also returned.
    """
    node_list = filter_nodes(nodes, middle_nodes=middle_nodes)
    return [node for node in node_list if function(node, *args, **kwargs)]


def get_min_max_nodes(nodes, *, middle_nodes=False):
    """Return a geometry set with the max and min nodes in all directions.

    Args
    ----
    nodes: list(Nodes)
        If this one is given return an array with the coordinates of the
        nodes in list, otherwise of all nodes in the mesh.
    middle_nodes: bool
        If this is true, middle nodes of a beam are also returned.
    """

    from .container import GeometryName

    node_list = filter_nodes(nodes, middle_nodes=middle_nodes)
    geometry = GeometryName()

    pos = get_nodal_coordinates(node_list)
    for i, direction in enumerate(["x", "y", "z"]):
        # Check if there is more than one value in dimension.
        min_max = [np.min(pos[:, i]), np.max(pos[:, i])]
        if np.abs(min_max[1] - min_max[0]) >= mpy.eps_pos:
            for j, text in enumerate(["min", "max"]):
                # get all nodes with the min / max coordinate
                min_max_nodes = []
                for index, value in enumerate(
                    np.abs(pos[:, i] - min_max[j]) < mpy.eps_pos
                ):
                    if value:
                        min_max_nodes.append(node_list[index])
                geometry[f"{direction}_{text}"] = GeometrySet(min_max_nodes)
    return geometry


def is_node_on_plane(
    node, *, normal=None, origin_distance=None, point_on_plane=None, tol=mpy.eps_pos
):
    """Query if a node lies on a plane defined by a point_on_plane or the
    origin distance.

    Args
    ----
    node:
        Check if this node coincides with the defined plane.
    normal: np.array, list
        Normal vector of defined plane.
    origin_distance: float
        Distance between origin and defined plane. Mutually exclusive with
        point_on_plane.
    point_on_plane: np.array, list
        Point on defined plane. Mutually exclusive with origin_distance.
    tol: float
        Tolerance of evaluation if point coincides with plane

    Return
    ----
    True if the point lies on the plane, False otherwise.
    """

    if origin_distance is None and point_on_plane is None:
        raise ValueError("Either provide origin_distance or point_on_plane!")
    elif origin_distance is not None and point_on_plane is not None:
        raise ValueError("Only provide origin_distance OR point_on_plane!")

    if origin_distance is not None:
        projection = np.dot(node.coordinates, normal) / np.linalg.norm(normal)
        distance = np.abs(projection - origin_distance)
    elif point_on_plane is not None:
        distance = np.abs(
            np.dot(point_on_plane - node.coordinates, normal) / np.linalg.norm(normal)
        )

    return distance < tol


def get_env_variable(name, *, default="default_not_set"):
    """Return the value of an environment variable.

    Args
    ----
    name: str
        Name of the environment variable
    default:
        Value to be returned if the given named environment variable does
        not exist. If this is not set and the name is not in the env
        variables, then an error will be thrown.
    """
    if name in os.environ.keys():
        return os.environ[name]
    elif default == "default_not_set":
        raise ValueError(f"Environment variable {name} is not set")
    return default
