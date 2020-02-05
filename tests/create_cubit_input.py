# -*- coding: utf-8 -*-
"""
This script creates a solid input files with cubitpy.
"""

# Python imports.
import os


# Cubitpy imports.
from cubitpy import CubitPy, cupy
from cubitpy.mesh_creation_functions import create_brick


def create_tube_cubit():
    """Create a solid tube used by the meshpy testing functions."""

    # Initialize cubit.
    cubit = CubitPy()

    # Set header.
    cubit.head = '''
        ------------------------------------------------------------PROBLEM TYP
        PROBLEMTYP                      Structure
        RESTART                         0
        ----------------------------------------------------------DISCRETISATION
        NUMFLUIDDIS                     0
        NUMSTRUCDIS                     1
        NUMALEDIS                       0
        NUMTHERMDIS                     0
        ---------------------------------------------------------------------IO
        OUTPUT_BIN                      Yes
        STRUCT_DISP                     Yes
        FILESTEPS                       1000
        VERBOSITY                       Standard
        --------------------------------------------------IO/RUNTIME VTK OUTPUT
        OUTPUT_DATA_FORMAT              binary
        INTERVAL_STEPS                  1
        EVERY_ITERATION                 No
        ----------------------------------------IO/RUNTIME VTK OUTPUT/STRUCTURE
        OUTPUT_STRUCTURE                Yes
        DISPLACEMENT                    Yes
        -----------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                   1
        INT_STRATEGY                    Standard
        DYNAMICTYP                      Statics
        RESULTSEVRY                     1
        RESTARTEVRY                     5
        NLNSOL                          fullnewton
        PREDICT                         TangDis
        TIMESTEP                        0.05
        NUMSTEP                         20
        MAXTIME                         1.0
        TOLRES                          1.0E-5
        TOLDISP                         1.0E-11
        NORM_RESF                       Abs
        NORM_DISP                       Abs
        NORMCOMBI_RESFDISP              And
        MAXITER                         20
        ---------------------------------------------------------------SOLVER 1
        NAME                            Structure_Solver
        SOLVER                          UMFPACK
        ------------------------------------------------------------------MATERIALS
        MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 1.0e+09 NUE 0.0 DENS 7.80E-6 THEXPANS 0.0
        ---------------------------------------------------------------------FUNCT1
        COMPONENT 0 FUNCTION cos(2*pi*t)
        -----------------------------------------------------------------FUNCT2
        COMPONENT 0 FUNCTION sin(2*pi*t)
        '''

    # Geometry parameters
    h = 10
    r = 0.25

    # Mesh parameters
    n_circumference = 6
    n_height = 10

    # Create cylinder.
    cylinder = cubit.cylinder(h, r, r, r)

    # Set the mesh size.
    for curve in cylinder.curves():
        cubit.set_line_interval(curve, n_circumference)
    cubit.cmd('surface 1 size {}'.format(h / n_height))

    # Mesh the geometry.
    cylinder.volumes()[0].mesh()

    # Set blocks and sets.
    cubit.add_element_type(cylinder.volumes()[0], cupy.element_type.hex8,
        name='tube')
    cubit.add_node_set(cylinder.surfaces()[1], name='fix',
        bc_section='DESIGN SURF DIRICH CONDITIONS',
        bc_description=(
            'NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
            + 'FUNCT 0 0 0 0 0 0'))
    cubit.add_node_set(cylinder.surfaces()[2], name='dirichlet_controlled',
        bc_section='DESIGN SURF DIRICH CONDITIONS',
        bc_description=(
            'NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 '
            + 'FUNCT 1 2 0 0 0 0'))

    # Return the cubit object.
    return cubit


def create_tube(file_path):
    """Write the solid tube to a file."""

    # Export mesh.
    create_tube_cubit().create_dat(file_path)


def create_block_cubit():
    """Create a solid block in cubit and add a volume condition."""

    # Initialize cubit.
    cubit = CubitPy()

    # Create the block.
    cube = create_brick(cubit, 1, 1, 1, mesh_factor=9)

    # Add the boundary condition.
    cubit.add_node_set(cube.volumes()[0],
        bc_type=cupy.bc_type.beam_to_solid_volume_meshtying,
        bc_description='COUPLING_ID 1')

    # Add the boundary condition.
    cubit.add_node_set(cube.surfaces()[0],
        bc_type=cupy.bc_type.beam_to_solid_surface_meshtying,
        bc_description='COUPLING_ID 2')

    # Set point coupling conditions.
    nodes = cubit.group()
    nodes.add([cube.vertices()[0], cube.vertices()[2]])
    cubit.add_node_set(
        nodes,
        bc_type=cupy.bc_type.point_coupling,
        bc_description='NUMDOF 3 ONOFF 1 2 3'
        )

    # Return the cubit object.
    return cubit


def create_block(file_path):
    """Create the solid cube in cubit and write it to a file."""

    # Export mesh.
    create_block_cubit().create_dat(file_path)


if __name__ == '__main__':
    # Execution part of script.

    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Create the input file for the solid tube.
    file_path = os.path.join(dir_path,
        'reference-files/baci_input_solid_tube.dat')
    create_tube(file_path)

    # Create the input files for the solid cube.
    file_path = os.path.join(dir_path,
        'reference-files/test_meshpy_btsvm_coupling_solid_mesh.dat')
    create_block(file_path)
