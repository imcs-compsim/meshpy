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
Generic function used to create all beams within meshpy.
"""

# Meshpy modules.
from ..conf import mpy
from ..container import GeometryName
from ..geometry_set import GeometrySet
from ..utility import get_single_node


def create_beam_mesh_function(
    mesh,
    *,
    beam_object=None,
    material=None,
    function_generator=None,
    interval=None,
    n_el=None,
    l_el=None,
    interval_length=None,
    add_sets=False,
    start_node=None,
    end_node=None,
    vtk_cell_data=None
):
    """
    Generic beam creation function.

    Args
    ----
    mesh: Mesh
        Mesh that the created beam(s) should be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    function_generator: function that returns function
        The function_generator has to take two variables, point_a and
        point_b (both within the interval) and return a function(xi) that
        calculates the position and rotation along the beam, where
        point_a -> xi = -1 and point_b -> xi = 1.
    interval: [start end]
        Start and end values for interval that will be used to create the
        beam.
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el
    l_el: float
        Desired length of beam elements. This requires the option interval_length
        to be set. Mutually exclusive with n_el. Be aware, that this length
        might not be achieved, if the elements are warped after they are
        created.
    interval_length:
        Total length of the interval. Is required when the option l_el is given.
    add_sets: bool
        If this is true the sets are added to the mesh and then displayed
        in eventual VTK output, even if they are not used for a boundary
        condition or coupling.
    start_node: Node, GeometrySet
        Node to use as the first node for this line. Use this if the line
        is connected to other lines (angles have to be the same, otherwise
        connections should be used). If a geometry set is given, it can
        contain one, and one node only. If the rotation does not match, but
        the tangent vector is the same, the created beams triads are
        rotated so the physical problem stays the same (for axi-symmetric
        beam cross-sections) but the same nodes can be used.
    end_node: Node, GeometrySet, bool
        If this is a Node or GeometrySet, the last node of the created beam
        is set to that node.
        If it is True the created beam is closed within itself.
    vtk_cell_data: {cell_data_name (str): cell_data_value (float)}
        With this argument, a vtk cell data can be set for the elements
        created within this function. This can be used to check which
        elements are created by which function.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the curve. Also a 'line' set
        with all nodes of the curve.
    """

    # Get the number of elements
    if n_el is None and l_el is None:
        n_el = 1
    elif n_el is not None and l_el is None:
        pass
    elif n_el is None and l_el is not None:
        if interval_length is None:
            raise ValueError("The parameters l_el requires interval_length to be set")
        n_el = max([1, round(interval_length / l_el)])
    else:
        raise ValueError("The parameters n_el and l_el are mutually exclusive")

    # Make sure the material is in the mesh.
    mesh.add_material(material)

    # List with nodes and elements that will be added in the creation of
    # this beam.
    elements = []
    nodes = []

    def check_given_node(node):
        """Check that the given node is already in the mesh."""
        if node not in mesh.nodes:
            raise ValueError("The given node is not in the current mesh")

    # If a start node is given, set this as the first node for this beam.
    if start_node is not None:
        start_node = get_single_node(start_node, check_cosserat_node=True)
        nodes = [start_node]
        check_given_node(start_node)

    # If an end node is given, check what behavior is wanted.
    close_beam = False
    if end_node is True:
        close_beam = True
    elif end_node is not None:
        end_node = get_single_node(end_node, check_cosserat_node=True)
        check_given_node(end_node)

    # Create the beams.
    for i in range(n_el):

        # If the beam is closed with itself, set the end node to be the
        # first node of the beam. This is done when the second element is
        # created, as the first node already exists here.
        if i == 1 and close_beam:
            end_node = nodes[0]

        # Get the function to create this beam element.
        function = function_generator(
            interval[0] + i * (interval[1] - interval[0]) / n_el,
            interval[0] + (i + 1) * (interval[1] - interval[0]) / n_el,
        )

        # Set the start node for the created beam.
        if start_node is not None or i > 0:
            first_node = nodes[-1]
        else:
            first_node = None

        # If an end node is given, set this one for the last element.
        if end_node is not None and i == n_el - 1:
            last_node = end_node
        else:
            last_node = None

        element = beam_object(material=material)
        elements.append(element)
        nodes.extend(
            element.create_beam(function, start_node=first_node, end_node=last_node)
        )

    # Set vtk cell data on created elements.
    if vtk_cell_data is not None:
        for data_name, data_value in vtk_cell_data.items():
            for element in elements:
                if data_name in element.vtk_cell_data.keys():
                    raise KeyError(
                        'The cell data "{}" already exists!'.format(data_name)
                    )
                element.vtk_cell_data[data_name] = data_value

    # Add items to the mesh
    mesh.elements.extend(elements)
    if start_node is None:
        mesh.nodes.extend(nodes)
    else:
        mesh.nodes.extend(nodes[1:])

    # Set the last node of the beam.
    if end_node is None:
        end_node = nodes[-1]

    # Set the nodes that are at the beginning and end of line (for search
    # of overlapping points)
    nodes[0].is_end_node = True
    end_node.is_end_node = True

    # Create geometry sets that will be returned.
    return_set = GeometryName()
    return_set["start"] = GeometrySet(nodes[0])
    return_set["end"] = GeometrySet(end_node)
    return_set["line"] = GeometrySet(elements)
    if add_sets:
        mesh.add(return_set)
    return return_set
