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
"""This script creates a solid input files with cubitpy."""

import os

from cubitpy import CubitPy, cupy
from cubitpy.mesh_creation_functions import create_brick, extrude_mesh_normal_to_surface

from meshpy import InputFile, mpy


def create_tube_cubit_mesh(r, h, n_circumference, n_height):
    """Create a solid tube in cubit.

    Args
    ----
    r: float
        Radius of the cylinder.
    h: float
        Height of the cylinder.
    n_circumference: int
        Number of elements along the circumferential direction.
    n_height: int
        Number of elements along the axial direction.

    Return
    ----
    The created cubit object.
    """

    # Initialize cubit.
    cubit = CubitPy()

    # Create cylinder.
    cylinder = cubit.cylinder(h, r, r, r)

    # Set the mesh size.
    for curve in cylinder.curves():
        cubit.set_line_interval(curve, n_circumference)
    cubit.cmd("surface 1 size {}".format(h / n_height))

    # Set blocks and sets.
    cubit.add_element_type(cylinder.volumes()[0], cupy.element_type.hex8, name="tube")

    # Return the cubit object.
    return cubit, cylinder


def create_tube_cubit():
    """Load the solid tube and add input file parameters."""

    # Initialize cubit.
    cubit, cylinder = create_tube_cubit_mesh(0.25, 10.0, 6, 10)

    # Mesh the geometry.
    cylinder.volumes()[0].mesh()

    # Set boundary conditions.
    cubit.add_node_set(
        cylinder.surfaces()[1],
        name="fix",
        bc_section="DESIGN SURF DIRICH CONDITIONS",
        bc_description=(
            "NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 0.0 0.0 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0 0 0"
        ),
    )
    cubit.add_node_set(
        cylinder.surfaces()[2],
        name="dirichlet_controlled",
        bc_section="DESIGN SURF DIRICH CONDITIONS",
        bc_description=(
            "NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0"
        ),
    )

    # Set header.
    cubit.head = """
        -----------------------------------------------------------PROBLEM TYPE
        PROBLEMTYPE                     Structure
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
        DYNAMICTYPE                     Statics
        RESULTSEVERY                    1
        RESTARTEVERY                    5
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
        MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 1.0e+09 NUE 0.0 DENS 7.80E-6
        ---------------------------------------------------------------------FUNCT1
        COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME cos(2*pi*t)
        -----------------------------------------------------------------FUNCT2
        COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME sin(2*pi*t)
        """

    # Return the cubit object.
    return cubit


def create_tube(file_path):
    """Write the solid tube to a file."""

    # Export mesh.
    create_tube_cubit().create_dat(file_path)


def create_tube_tutorial(file_path):
    """Create the solid tube for the tutorial."""

    # Initialize cubit.
    cubit, cylinder = create_tube_cubit_mesh(0.05, 3.0, 6, 10)

    # Put the tube in the correct position.
    cubit.cmd("rotate volume 1 angle -90 about X include_merged")
    cubit.move(cylinder, [0, 1.5, 1.5])

    # Mesh the geometry.
    cylinder.volumes()[0].mesh()

    # Set boundary conditions.
    cubit.add_node_set(
        cylinder.surfaces()[1],
        name="fix",
        bc_type=cupy.bc_type.dirichlet,
        bc_description="NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0",
    )
    cubit.add_node_set(
        cylinder.surfaces()[2],
        name="dirichlet_controlled",
        bc_type=cupy.bc_type.dirichlet,
        bc_description="NUMDOF 3 ONOFF 1 0 0 VAL 0.5 0 0 FUNCT 1 0 0",
    )

    # Set header.
    cubit.head = """
        ------------------------------------------------------------------MATERIALS
        MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 1.0 NUE 0 DENS 0
        """

    # Export mesh.
    cubit.create_dat(file_path)


def create_block_cubit():
    """Create a solid block in cubit and add a volume condition."""

    # Initialize cubit.
    cubit = CubitPy()

    # Create the block.
    cube = create_brick(cubit, 1, 1, 1, mesh_factor=9)

    # Add the boundary condition.
    cubit.add_node_set(
        cube.volumes()[0],
        bc_type=cupy.bc_type.beam_to_solid_volume_meshtying,
        bc_description="COUPLING_ID 1",
    )

    # Add the boundary condition.
    cubit.add_node_set(
        cube.surfaces()[0],
        bc_type=cupy.bc_type.beam_to_solid_surface_meshtying,
        bc_description="COUPLING_ID 2",
    )

    # Set point coupling conditions.
    nodes = cubit.group()
    nodes.add([cube.vertices()[0], cube.vertices()[2]])
    cubit.add_node_set(
        nodes,
        bc_type=cupy.bc_type.point_coupling,
        bc_description="NUMDOF 3 ONOFF 1 2 3",
    )

    # Return the cubit object.
    return cubit


def create_block(file_path):
    """Create the solid cube in cubit and write it to a file."""

    # Export mesh.
    create_block_cubit().create_dat(file_path)


def create_solid_shell_meshes(file_path_blocks, file_path_dome):
    """Create the meshes needed for the solid shell tests."""

    def create_brick_mesh(
        dimensions, n_elements, *, element_type=cupy.element_type.hex8sh
    ):
        """Create a MeshPy mesh with a solid brick."""
        cubit = CubitPy()
        create_brick(
            cubit,
            *dimensions,
            mesh_interval=n_elements,
            element_type=element_type,
            mesh=True,
        )
        mpy.import_mesh_full = True
        return InputFile(cubit=cubit)

    # Create the input file with the blocks representing plates in different planes
    mesh = InputFile()
    dimensions = [0.1, 2, 4]
    elements = [1, 2, 2]

    def rotate_list(original_list, n):
        """Rotate the list."""
        return original_list[-n:] + original_list[:-n]

    # Add the plates in all directions (permute the dimension and number of elements
    # in each direction)
    for i in range(3):
        brick = create_brick_mesh(rotate_list(dimensions, i), rotate_list(elements, i))
        brick.translate([i * 4, 0, 0])
        mesh.add(brick)

    # Add a last plate with standard solid elements, to make sure that the algorithm
    # skips those
    brick = create_brick_mesh(
        rotate_list(dimensions, 1),
        rotate_list(elements, 1),
        element_type=cupy.element_type.hex8,
    )
    brick.translate([3 * 4, 0, 0])
    mesh.add(brick)

    mesh.write_input_file(file_path_blocks, header=False)

    # Create the dome input
    cubit = CubitPy()
    cubit.cmd("create sphere radius 1 zpositive")
    cubit.cmd("surface 2 size auto factor 6")
    cubit.cmd("mesh surface 2")
    dome_mesh = extrude_mesh_normal_to_surface(
        cubit, [cubit.surface(2)], 0.1, n_layer=1
    )
    cubit.add_element_type(dome_mesh, cupy.element_type.hex8sh)
    cubit.create_dat(file_path_dome)


if __name__ == "__main__":
    # Execution part of script.

    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Create the input file for the solid tube.
    file_path = os.path.join(dir_path, "reference-files/4C_input_solid_tube.dat")
    create_tube(file_path)

    # Create the input files for the solid cube.
    file_path = os.path.join(
        dir_path, "reference-files/test_meshpy_btsvm_coupling_solid_mesh.dat"
    )
    create_block(file_path)

    # Create the tube for the tutorial.
    file_path = os.path.join(
        os.path.dirname(dir_path), "tutorial/4C_input_solid_tutorial.dat"
    )
    create_tube_tutorial(file_path)

    # Create the input files for the solid shell direction testing
    file_path_blocks = os.path.join(
        dir_path, "reference-files/4C_input_solid_shell_thickness_blocks.dat"
    )
    file_path_dome = os.path.join(
        dir_path, "reference-files/4C_input_solid_shell_thickness_dome.dat"
    )
    create_solid_shell_meshes(file_path_blocks, file_path_dome)
