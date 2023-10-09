# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2023
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
This script contains a tutorial for meshpy. Most basic functionality is covered
by this tutorial. For more information have a closer look at the test cases,
as they cover all functionality.
"""

# Import python modules.
import numpy as np
import autograd.numpy as npAD
import os

# Import the objects we need from meshpy.
from meshpy import (
    mpy,
    Mesh,
    MaterialReissner,
    Beam3rHerm2Line3,
    BoundaryCondition,
    Rotation,
    Function,
    GeometrySet,
    InputFile,
)
from meshpy.utility import get_single_node
from meshpy.mesh_creation_functions import (
    create_beam_mesh_line,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_curve,
)


def meshpy_tutorial(base_dir, preview=False):
    """
    Create a honeycomb like structure with different type of connectors.

    Args
    ----
    base_dir: str
        Path where all created files will be saved.
    """

    # In the first step an empty Mesh object is created, which in general holds
    # information about the nodes, elements, materials and boundary conditions.
    # A mesh can be added to a mesh, i.e. created geometries can be combined.
    # The first geometry is created in the final mesh, as its position is
    # already final.
    mesh = Mesh()

    # We now add a straight line, with a SR beam element. If other beam
    # theories or element orders are used, simply replace the material and beam
    # objects.
    # The line is created between the two given points, with n_el elements.
    # Each mesh creation function returns certain geometry sets, where boundary
    # conditions can be applied or points can be coupled.
    mat = MaterialReissner(youngs_modulus=1.0, radius=0.02)
    beam_object = Beam3rHerm2Line3
    beam_set_1 = create_beam_mesh_line(
        mesh, beam_object, mat, [0, 0, 0], [0, 0.5, 0], n_el=5
    )

    # We now add a second line, extending the first one and we give the end
    # node of the first line as a parameter, then the new line uses the end
    # node of the last line as a starting node, i.e. the lines are connected.
    # This only works if the two nodes have the same position and orientation,
    # i.e. corners have to be coupled via coupling conditions.
    create_beam_mesh_line(
        mesh,
        beam_object,
        mat,
        [0, 0.5, 0],
        [0, 1.0, 0],
        n_el=1,
        start_node=beam_set_1["end"],
    )

    # The mesh should now have 13 nodes and 6 elements.
    # Note that the employed SR elements insert a middle node in each element.
    print("mesh_1 number of nodes: {}".format(len(mesh.nodes)))
    print("mesh_1 number of elements: {}".format(len(mesh.elements)))

    # We can also look at the created mesh, either in python or in ParaView.
    if preview:
        mesh.display_python()
    mesh.write_vtk("step_1", base_dir)

    # We want to fix all positions and rotations of the first node.
    # mpy is a global object that stores enums and other options for meshpy.
    mesh.add(
        BoundaryCondition(
            beam_set_1["start"],
            (
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 "
                + "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            bc_type=mpy.bc.dirichlet,
        )
    )

    # In the next few steps we will create the honeycomb structure. Therefore,
    # we create a honeycomb mesh object.
    mesh_honeycomb = Mesh()

    # Now lets add a new mesh and create a circular segment.
    mesh_arc = Mesh()
    beam_set_arc = create_beam_mesh_arc_segment_2d(
        mesh_arc, beam_object, mat, [0, 0, 0], 1, 0, np.pi / 3.0, n_el=3
    )

    # Opening it in ParaView, will show the arc.
    mesh_arc.write_vtk("step_2", base_dir)

    # Now the arc is moved, so it coincides with the end point of the last
    # line. Therefore, we get the positions of the two nodes.
    # This can be done with the utility function get_node.
    arc_start_point = get_single_node(beam_set_arc["start"]).coordinates
    line_end_point = mesh.nodes[-1].coordinates
    distance = line_end_point - arc_start_point
    mesh_arc.translate(distance)

    # The arc matches the end of the line.
    mesh_arc.write_vtk("step_3", base_dir)

    # The arc is also rotated.
    mesh_arc.rotate(Rotation([0, 0, 1], np.pi / 6.0), origin=line_end_point)
    mesh_arc.write_vtk("step_4", base_dir)

    # Finally, the mesh is added to the honeycomb mesh.
    mesh_honeycomb.add(mesh_arc)

    # Next, a sinusoidal beam is created.
    def beam_sinus(t):
        """
        Define a parametric function for the beam. For this the numpy wrapper
        autograd has to be used.
        """
        return 0.5 / npAD.pi * npAD.array([t, npAD.sin(t)])

    mesh_sin = Mesh()
    beam_set_sin = create_beam_mesh_curve(
        mesh_sin, beam_object, mat, beam_sinus, [0, 2.0 * np.pi], n_el=10
    )
    mesh_sin.write_vtk("step_5", base_dir)

    # The sinus is not also moved to the end of the initially created line and
    # rotated.
    sin_start_point = get_single_node(beam_set_sin["start"]).coordinates
    distance = line_end_point - sin_start_point
    mesh_sin.translate(distance)
    mesh_sin.rotate(Rotation([0, 0, 1], np.pi / 6.0), origin=line_end_point)
    mesh_sin.write_vtk("step_6", base_dir)

    # Also add this mesh to the honeycomb mesh.
    mesh_honeycomb.add(mesh_sin)

    # In a next step, a straight vertical load is added directly to the end of
    # the arc.
    start = get_single_node(beam_set_arc["end"]).coordinates
    create_beam_mesh_line(
        mesh_honeycomb, beam_object, mat, start, start + [0, 1, 0], n_el=1
    )

    # Half of the honeycomb is now complete.
    mesh_honeycomb.write_vtk("step_7", base_dir)

    # We now create a copy of the honeycomb and reflect it, so it is a complete
    # honeycomb structure.
    mesh_honeycomb_copy = mesh_honeycomb.copy()
    mesh_honeycomb_copy.reflect(
        [np.sin(np.pi / 6.0), np.cos(np.pi / 6.0), 0],
        get_single_node(beam_set_sin["end"]).coordinates,
    )
    mesh_honeycomb_copy.write_vtk("step_8", base_dir)

    # Add the reflected part and the honeycomb like structure is complete.
    mesh_honeycomb.add(mesh_honeycomb_copy)

    # Now add everything to the final mesh.
    mesh.add(mesh_honeycomb)
    mesh.write_vtk("step_9", base_dir)

    # The mesh is now complete, but the nodes at the corners are not connected.
    # This can be done easily with couple_nodes.
    print(
        'Point couplings before "couple_nodes": {}'.format(
            len(mesh.boundary_conditions[(mpy.bc.point_coupling, mpy.geo.point)])
        )
    )
    mesh.couple_nodes()
    print(
        'Point couplings after "couple_nodes": {}'.format(
            len(mesh.boundary_conditions[(mpy.bc.point_coupling, mpy.geo.point)])
        )
    )

    # We can also display the coupling points in the vtk output.
    mesh.write_vtk("step_10", base_dir, coupling_sets=True)

    # Finally the mesh is wrapped around a cylinder. Therefore, the mesh is
    # has to be moved to the correct position. For details look into the
    # documentation of wrap_around_cylinder.
    mesh.rotate(Rotation([0, 1, 0], 0.5 * np.pi))
    mesh.translate([1, 0, 0])
    mesh.write_vtk("step_11", base_dir)

    mesh.wrap_around_cylinder()
    mesh.write_vtk("step_12", base_dir)

    # The geometry of the mesh is now complete.
    # We add a line load in y-direction to all beam elements. The line load
    # is controlled by the function fun_t.
    line_load_val = 0.00000001
    fun_t = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
    mesh.add(fun_t)
    geometry_set_all_lines = GeometrySet(mesh.elements)
    mesh.add(
        BoundaryCondition(
            geometry_set_all_lines,
            (
                "NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 "
                + "FUNCT 0 {} 0 0 0 0 0 0 0"
            ),
            format_replacement=[line_load_val, fun_t],
            bc_type=mpy.bc.neumann,
        )
    )

    # The vtk output will also show all node sets for BCs on the mesh.
    mesh.write_vtk("step_13", base_dir)

    # The object InputFile is a mesh, but can also store BACI input parameters.
    # Additionally we load an existing solid mesh. This shows how solid, or in
    # general, volume elements (fluid, ...) can be combined with beam elements.
    # Everything from the volume input file will be included in the combined
    # input file, e.g. BC, loads, materials, solver parameters, ... .
    solid_dat_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "tests",
        "reference-files",
        "baci_input_solid_tutorial.dat",
    )
    input_file = InputFile(dat_file=solid_dat_path)

    # Add the beam geometry to the input file.
    input_file.add(mesh)

    # Add the input parameters.
    input_file.add(
        """
        ------------------------------------------------------------------TITLE
        meshpy tutorial
        ------------------------------------------------------------PROBLEM TYP
        PROBLEMTYP                            Structure
        RESTART                               0
        ---------------------------------------------------------------------IO
        OUTPUT_BIN                            no
        STRUCT_DISP                           yes
        FILESTEPS                             1000
        VERBOSITY                             Standard
        STRUCT_STRAIN                         yes
        STRUCT_STRESS                         yes
        -----------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                         1
        INT_STRATEGY                          Standard
        DYNAMICTYP                            Statics
        RESULTSEVRY                           1
        NLNSOL                                fullnewton
        TIMESTEP                              0.1
        NUMSTEP                               10
        MAXTIME                               1.0
        ---------------------------------------------------------------SOLVER 1
        NAME                                  Structure_Solver
        SOLVER                                Superlu
        --------------------------------------------------IO/RUNTIME VTK OUTPUT
        OUTPUT_DATA_FORMAT                    binary
        INTERVAL_STEPS                        1
        EVERY_ITERATION                       no
        ----------------------------------------IO/RUNTIME VTK OUTPUT/STRUCTURE
        OUTPUT_STRUCTURE                      yes
        DISPLACEMENT                          yes
        --------------------------------------------IO/RUNTIME VTK OUTPUT/BEAMS
        OUTPUT_BEAMS                          yes
        DISPLACEMENT                          yes
        USE_ABSOLUTE_POSITIONS                yes
        TRIAD_VISUALIZATIONPOINT              yes
        STRAINS_GAUSSPOINT                    yes
        """
    )

    return input_file


if __name__ == "__main__":
    """Execution part of script."""

    # Adapt this path to the directory you want to store the tutorial files in.
    tutorial_directory = None
    input_file = meshpy_tutorial(tutorial_directory)
    input_file.write_input_file(os.path.join(tutorial_directory, "tutorial.dat"))
