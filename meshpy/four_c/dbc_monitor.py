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
from .. import mpy, GeometrySet, BoundaryCondition,Function
from .. utility import linear_transformations
from meshpy.function_utility import create_linear_interpolation_function

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

    nodes, time, force, _ = read_dbc_monitor_file(file_path)

    # The forces are the negative reactions at the Dirichlet boundaries.
    force *= -1.0
    fun_array=[]
    time_str=' '.join(map(str, np.insert(time+dt,0,0,axis=0)))
    force=np.append(force[-1],np.flipud (force)).reshape(force.shape[0]+1,force.shape[1])
    if function==False:
        for dim in range(force.shape[1]):
            # Extract the first element from each sub-array
            first_elements = [force_per_direction[dim] for force_per_direction in force]

            # Convert the list of first elements to a string
            first_elements_str = ' '.join(map(str, first_elements))

            fun=Function(
            """SYMBOLIC_FUNCTION_OF_TIME a \nVARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS """+str(force.shape[0])+
            " TIMES "+time_str+" VALUES "+first_elements_str
            )
            fun_array.append(str(len(input_file.functions)+1))
            input_file.add(fun)

        # not set forces to 1 since the force values are extracted into a function
        force=0*force+1
    else:
        fun_array=np.array([1, 1, 1])*function
    # Create the BC condition for this set and add it to the input file.
    mesh_nodes = [input_file.nodes[i_node] for i_node in nodes]
    geo = GeometrySet(mesh_nodes)
    extra_dof_zero = " 0" * (n_dof - 3)
    bc = BoundaryCondition(
        geo,
        (
            "NUMDOF {n_dof} ONOFF 1 1 1{edz} VAL {data[0]} {data[1]} {data[2]}"
            "{edz} FUNCT {function[0]} {function[1]} {function[2]}{edz}"
        ).format(n_dof=n_dof, data=force[step], edz=extra_dof_zero,function=fun_array),
        bc_type=mpy.bc.neumann
    )
    input_file.add(bc)



def dbc_monitor_to_input_with_interpolation(input_file, file_path, step=-1, n_dof=3,timespan=[0,1,100],type="",flip=False):
    """
    Converts the Dirichlet boundary condition monitor log to a Neumann
    boundary condition input section by using interpolation between all calculated
    function values

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
        Indicates until which step the values are used. Default is -1, i.e. the last step.
    n_dof: int
        Number of DOFs per node.
    timespan: [t1, t2, t3] in float
        transforms the given time array into this specific format
        Additionally the time array always starts at 0 and ends at t3 to ensure a valid simulation
    type of lin: interpolatoion
        two types:
            1) not specified simple interpolation between t1 and t2
            2) type: "hat" stacks puts the values first and then decays them
            interpolation starting from t1 going to a max at (t1+t2)/2 and going back to the value at t1
    flip: bool
        indicates, if the resulting forces are flipped in the array
         in case of flip, the final forces are applied at t_start.


    """

    if not len(timespan)==3:
        raise ValueError(f"Please provide a timeintverall with size 1x3 not {len(timespan)}")

    nodes, time, force, _ = read_dbc_monitor_file(file_path)

    # The forces are the negative reactions at the Dirichlet boundaries.
    force *= -1.0
    time=time[0:step]
    force=force[0:step,:]
    print("time",time)
    # apply transformations to time and forces so that the
    # 4c boundary conditions can work with them
    if type=="":
        time, force = linear_transformations(time,force,timespan,flip)


    print(force)

    if type=="hat":
        # create a hat function with peak locatet at timespan/2
        t_half=(timespan[1]-timespan[0])/2

        time1, force1 = linear_transformations(time,force,[0,1,1],flip)
        time2, force2 = linear_transformations(time,force,[0,1,timespan[2]],not flip)

        # remove first element since it is dublicated zero
        print(time1)
        print(time2)

        np.delete(time2,0)
        np.delete(force2,0)



        time = np.concatenate((time1, time2+max(time1)))
        print(time)
        force = np.concatenate((force1, force2), axis=0)


    elif type != "":
        raise ValueError("Your type is currently not supported.")

    # set up function array contianing all the functions which will be added to the input file
    fun_array=[]


    for dim in range(force.shape[1]):

        # Extract the elements of each dimension
        forces_per_dimension = [force_per_direction[dim] for force_per_direction in force]

        # create a linear function with the force values
        fun=create_linear_interpolation_function(time,forces_per_dimension, function_type="SYMBOLIC_FUNCTION_OF_TIME")

        # append array for some reason the funciton str(fun) is not working here...
        # current hack
        fun_array.append(str(len(input_file.functions)+1))

        # add the function to the input array
        input_file.add(fun)

    # now set forces to 1 since the force values are extracted already in the function
    force=0*force+1

    # Create the BC condition for this set and add it to the input file.
    mesh_nodes = [input_file.nodes[i_node] for i_node in nodes]
    geo = GeometrySet(mesh_nodes)
    extra_dof_zero = " 0" * (n_dof - 3)
    bc = BoundaryCondition(
        geo,
        (
            "NUMDOF {n_dof} ONOFF 1 1 1{edz} VAL {data[0]} {data[1]} {data[2]}"
            "{edz} FUNCT {function[0]} {function[1]} {function[2]}{edz}"
        ).format(n_dof=n_dof, data=force[step], edz=extra_dof_zero,function=fun_array),
        bc_type=mpy.bc.neumann
    )
    input_file.add(bc)
