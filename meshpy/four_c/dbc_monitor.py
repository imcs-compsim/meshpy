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
from .. import mpy, GeometrySet, BoundaryCondition, Function
from meshpy.function_utility import (
    create_linear_interpolation_function,
    linear_time_transformation,
)


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
    boundary condition input section. Uses only the last force value.

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


def all_dbc_monitor_values_to_input(
    input_file,
    file_path,
    steps=None,
    n_dof=3,
    time_span=[0, 1, 2],
    type=None,
    flip_forces=False,
    fun_array=[],
):
    """Extracts all the force values of the monitored Dirichlet Boundary Condition and converts
    them into a Function with a Neumann Boundary Condition for the input_file.
    The Monitor log force values must be obtained from a previous simulation with constant step size.
    The discretization of the previous simulation must be identical to the one within the input_file.
    The extracted force values are passed to a linear interpolation 4C-function.
    It is advisable to only call this function once all nodes have been added to the input file.

    Args
    ----
    input_file: InputFile
        The input file where the created Neumann boundary condition is added
        to. The nodes(eg. discretization) referred to in the log file must match with the ones
        in input_file.
    file_path: str
        Path to the Dirichlet boundary condition log file.
    steps: [int,int]
        Index range of which the force values are extracted. Default 0 and -1 extracts every point from the array.
    n_dof: int
        Number of DOFs per node.
    time_span: [t1, t2, t3] in float
        transforms the given time array into this specific format.
        The time array always starts at 0 and ends at t3 to ensure a valid simulation
    type: str or None
        two types are available:
            1) None: not specified simple extract all values and apply them between time intervall t1 and t2
            2) "hat": puts the values first until last value is reached and then decays them back to first value.
            interpolation starting from t1 going to the last value at (t1+t2)/2 and going back to the value at time t2
    flip_forces: bool
        indicates, if the extracted forces should be flipped or rearanged wrt. to the time
        For flip_forces=true, the forces at the final time are applied at t_start.
    fun_array: [Function, Function, Function]
        array consisting of 3 custom functions(x,y,z). The value for boundary condition is selected from the last steps.
    """

    nodes, time, force, _ = read_dbc_monitor_file(file_path)

    # The forces are the negative reactions at the Dirichlet boundaries.
    force *= -1.0

    # if special index range is provided use it
    if steps:
        time = time[steps[0] : steps[1] + 1]
        force = force[steps[0] : steps[1] + 1, :]
    else:
        # otherwise define steps from start to end
        steps = [0, -1]

    # apply transformations to time and forces according to the schema
    if type is None:
        if not len(time_span) == 2:
            raise ValueError(
                f"Please provide a time_span with size 1x2 not {len(time_span)}"
            )

        time, force = linear_time_transformation(time, force, time_span, flip_forces)
        if len(fun_array) != 3:
            print("Please provide a list with three valid Functions.")

    elif type == "hat":

        if not len(time_span) == 3:
            raise ValueError(
                f"Please provide a time_span with size 1x3 not {len(time_span)}"
            )

        if len(fun_array) > 0:
            print(
                "You selected type",
                type,
                ", however the provided functions ",
                fun_array,
                " are overwritten.",
            )
        fun_array = []

        # create the two intervals
        time1, force1 = linear_time_transformation(
            time, force, time_span[0:2], flip_forces
        )
        time2, force2 = linear_time_transformation(
            time, force, time_span[1:3], not flip_forces
        )

        # remove first element since it is duplicated zero
        np.delete(time2, 0)
        np.delete(force2, 0)

        # add the respective force
        time = np.concatenate((time1, time2[2:]))
        force = np.concatenate((force1, force2[2:]), axis=0)

    else:
        raise ValueError(
            "The selected type: "
            + type
            + " is currently not supported. Feel free to add it here."
        )

    # overwrite the function, if one is provided since for the specific types the function is generated
    if type:

        for dim in range(force.shape[1]):

            # Extract the elements of each dimension
            forces_per_dimension = [
                force_per_direction[dim] for force_per_direction in force
            ]

            # create a linear function with the force values
            fun = create_linear_interpolation_function(
                time, forces_per_dimension, function_type="SYMBOLIC_FUNCTION_OF_TIME"
            )

            # add the function to the input array
            input_file.add(fun)

            # store the id of the function
            # fun_array.append(str(len(input_file.functions)))
            fun_array.append(fun)

        # now set forces to 1 since the force values are already extracted in the function's values
        force = 0 * force + 1

    elif len(fun_array) != 3:
        raise ValueError("Please provide fun_array with ")

    # Create the BC condition for this set and add it to the input file.
    mesh_nodes = [input_file.nodes[i_node] for i_node in nodes]
    geo = GeometrySet(mesh_nodes)
    extra_dof_zero = " 0" * (n_dof - 3)
    bc = BoundaryCondition(
        geo,
        (
            "NUMDOF {n_dof} ONOFF 1 1 1{edz} VAL {data[0]} {data[1]} {data[2]}"
            "{edz} FUNCT {{}} {{}} {{}}{edz}"
        ).format(n_dof=n_dof, data=force[steps[1]], edz=extra_dof_zero),
        bc_type=mpy.bc.neumann,
        format_replacement=fun_array,
    )
    input_file.add(bc)
