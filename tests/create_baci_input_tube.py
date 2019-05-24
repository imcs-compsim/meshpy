# -*- coding: utf-8 -*-
"""
This script creates a tube for the meshpy test case.
"""

# Python imports.
import os


# Cubitpy imports.
from cubitpy import CubitPy


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
    cubit.add_element_type(cylinder.volumes()[0], 'HEX8', name='tube', bc=[
        'STRUCTURE',
        'MAT 1 KINEM nonlinear EAS none',
        'SOLIDH8'
        ])
    cubit.add_node_set(cylinder.surfaces()[1], name='fix', bc=[
        'DESIGN SURF DIRICH CONDITIONS',
        ('NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
            + 'FUNCT 0 0 0 0 0 0')])
    cubit.add_node_set(cylinder.surfaces()[2], name='dirichlet_controlled',
        bc=[
            'DESIGN SURF DIRICH CONDITIONS',
            ('NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 '
                + 'FUNCT 1 2 0 0 0 0')])

    # Return the cubit object.
    return cubit


def create_tube(file_path):
    """Write the solid tube to a file."""

    # Export mesh.
    create_tube_cubit().create_dat(file_path)


if __name__ == '__main__':
    # Execution part of script.

    dir_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(dir_path, 'testing-tmp/baci_input_tube.dat')
    create_tube(file_path)
