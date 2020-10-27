# -*- coding: utf-8 -*-
"""
This function converts the DBC monitor log files to Neumann input sections.
"""


# Python modules.
import numpy as np

# Meshpy stuff.
from .. import mpy, GeometrySet, BoundaryCondition


def dbc_monitor_to_input(input_file, file_path, step=-1, function=1, n_dof=3):
    """
    Convert the Dirichlet boundary condition monitor log to a Neumann
    boundary condition input section.

    Args
    ----
    input_file: InputFile
        The input file where the created Neumann boundary condition is added
        to. The nodes refered to in the log file have to match with the ones
        in the input section. It is advisable to only call this cuntion once
        all nodes have been added to the input file.
    file_path: str
        Path to the Dirichlet boundary condition log file.
    step: int
        Step values to be used. Default is -1, i.e. the last step.
    function: Function, int
        Function for the Neuman boundary condition.
    n_dof: int
        Number of DOFs per node.
    """

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Extract the nodes for this condition.
    node_line = lines[0].split(' ')
    counter = node_line.index(':') + 1
    nodes = []
    is_int = True
    while is_int:
        try:
            node_id = int(node_line[counter])
            nodes.append(node_id)
        except ValueError:
            is_int = False
        counter += 1

    # Find the start of the data lines.
    for i, line in enumerate(lines):
        if line.split(' ')[0] == 'step':
            break
    else:
        raise ValueError('Could not find "step" in file!')
    start_line = i + 1

    # Get the monitor data.
    data = []
    for line in lines[start_line:]:
        data.append(np.fromstring(line, dtype=float, sep=' '))
    data = np.array(data)

    # The forces are the negative reactions at the Dirichlet boundaries.
    data_force = -data[:, -3:]

    # Create the BC condition for this set and add it to the input file.
    mesh_nodes = [input_file.nodes[i_node] for i_node in nodes]
    geo = GeometrySet(mpy.geo.point, nodes=mesh_nodes)
    extra_dof_zero = ' 0' * (n_dof - 3)
    bc = BoundaryCondition(geo,
        ('NUMDOF 6 ONOFF 1 1 1{edz} VAL {data[0]} {data[1]} {data[2]}{edz}'
        + ' FUNCT {{0}} {{0}} {{0}}{edz}').format(
            data=data_force[step], edz=extra_dof_zero),
        bc_type=mpy.bc.neumann,
        format_replacement=[function]
        )
    input_file.add(bc)
