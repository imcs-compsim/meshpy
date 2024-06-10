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
"""
This function converts the DBC monitor log files to Neumann input sections.
"""


# Python modules.
import numpy as np

# Meshpy stuff.
from .. import mpy, GeometrySet, BoundaryCondition


def read_dbc_monitor_file(file_path):
    """
    Load the Dirichlet boundary condition monitor log and return the data as
    well as the nodes of this boundary condition.

    Args
    ----
    file_path: str
        Path to the Dirichlet boundary condition monitor log.

    Return
    ----
    [node_ids], [time], [force], [moment]
    """

    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Extract the nodes for this condition.
    condition_nodes_prefix = "Nodes of this condition:"
    if not lines[1].startswith(condition_nodes_prefix):
        raise ValueError(
            f'The second line in the monitor file is supposed to start with "{condition_nodes_prefix}" but the line reads "{lines[1]}"'
        )
    node_line = lines[1].split(":")[1]
    node_ids_str = node_line.split()
    nodes = []
    for node_id_str in node_ids_str:
        node_id = int(node_id_str)
        nodes.append(node_id)

    # Find the start of the data lines.
    for i, line in enumerate(lines):
        if line.split(" ")[0] == "step":
            break
    else:
        raise ValueError('Could not find "step" in file!')
    start_line = i + 1

    # Get the monitor data.
    data = []
    for line in lines[start_line:]:
        data.append(np.fromstring(line, dtype=float, sep=" "))
    data = np.array(data)

    return nodes, data[:, 1], data[:, 4:7], data[:, 7:]


def dbc_monitor_to_input(input_file, file_path, step=-1, function=1, n_dof=3):
    """
    Convert the Dirichlet boundary condition monitor log to a Neumann
    boundary condition input section.

    Args
    ----
    input_file: InputFile
        The input file where the created Neumann boundary condition is added
        to. The nodes referred to in the log file have to match with the ones
        in the input section. It is advisable to only call this function once
        all nodes have been added to the input file.
    file_path: str
        Path to the Dirichlet boundary condition log file.
    step: int
        Step values to be used. Default is -1, i.e. the last step.
    function: Function, int
        Function for the Neumann boundary condition.
    n_dof: int
        Number of DOFs per node.
    """

    nodes, _, force, _ = read_dbc_monitor_file(file_path)

    # The forces are the negative reactions at the Dirichlet boundaries.
    force *= -1.0

    # Create the BC condition for this set and add it to the input file.
    mesh_nodes = [input_file.nodes[i_node] for i_node in nodes]
    geo = GeometrySet(mesh_nodes)
    extra_dof_zero = " 0" * (n_dof - 3)
    bc = BoundaryCondition(
        geo,
        (
            "NUMDOF {n_dof} ONOFF 1 1 1{edz} VAL {data[0]} {data[1]} {data[2]}"
            "{edz} FUNCT {{0}} {{0}} {{0}}{edz}"
        ).format(n_dof=n_dof, data=force[step], edz=extra_dof_zero),
        bc_type=mpy.bc.neumann,
        format_replacement=[function],
    )
    input_file.add(bc)
