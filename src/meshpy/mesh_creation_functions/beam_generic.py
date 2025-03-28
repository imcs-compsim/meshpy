# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Generic function used to create all beams within meshpy."""

from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union

import numpy as _np

from meshpy.core.conf import mpy as _mpy
from meshpy.core.element_beam import Beam as _Beam
from meshpy.core.geometry_set import GeometryName as _GeometryName
from meshpy.core.geometry_set import GeometrySet as _GeometrySet
from meshpy.core.material import MaterialBeam as _MaterialBeam
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.node import NodeCosserat as _NodeCosserat
from meshpy.utils.nodes import get_single_node as _get_single_node


def create_beam_mesh_function(
    mesh: _Mesh,
    *,
    beam_object: _Type[_Beam],
    material: _MaterialBeam,
    function_generator: _Callable,
    interval: _Tuple[float, float],
    n_el: _Optional[int] = None,
    l_el: _Optional[float] = None,
    interval_length: _Optional[float] = None,
    node_positions_of_elements: _Optional[_List[float]] = None,
    start_node: _Optional[_Union[_NodeCosserat, _GeometrySet]] = None,
    end_node: _Optional[_Union[_NodeCosserat, _GeometrySet]] = None,
    close_beam: _Optional[bool] = False,
    vtk_cell_data: _Optional[_Dict[str, _Tuple]] = None,
) -> _GeometryName:
    """Generic beam creation function.

    Remark for given start and/or end nodes:
        If the rotation does not match, but the tangent vector is the same,
        the created beams triads are rotated so the physical problem stays
        the same (for axi-symmetric beam cross-sections) but the nodes can
        be reused.

    Args:
        mesh:
            Mesh that the created beam(s) should be added to.
        beam_object:
            Class of beam that will be used for this line.
        material:
            Material for this line.
        function_generator:
            The function_generator has to take two variables, point_a and
            point_b (both within the interval) and return a function(xi) that
            calculates the position and rotation along the beam, where
            point_a -> xi = -1 and point_b -> xi = 1.
        interval:
            Start and end values for interval that will be used to create the
            beam.
        n_el:
            Number of equally spaced beam elements along the line. Defaults to 1.
            Mutually exclusive with l_el
        l_el:
            Desired length of beam elements. This requires the option interval_length
            to be set. Mutually exclusive with n_el. Be aware, that this length
            might not be achieved, if the elements are warped after they are
            created.
        interval_length:
            Total length of the interval. Is required when the option l_el is given.
        node_positions_of_elements:
            A list of normalized positions (within [0,1] and in ascending order)
            that define the boundaries of beam elements along the created curve.
            The given values will be mapped to the actual `interval` given as an
            argument to this function. These values specify where elements start
            and end, additional internal nodes (such as midpoints in higher-order
            elements) may be placed automatically.
        start_node:
            Node to use as the first node for this line. Use this if the line
            is connected to other lines (angles have to be the same, otherwise
            connections should be used). If a geometry set is given, it can
            contain one, and one node only.
        end_node:
            If this is a Node or GeometrySet, the last node of the created beam
            is set to that node.
            If it is True the created beam is closed within itself.
        close_beam:
            If it is True the created beam is closed within itself (mutually
            exclusive with end_node).
        vtk_cell_data:
            With this argument, a vtk cell data can be set for the elements
            created within this function. This can be used to check which
            elements are created by which function.

    Returns:
        Geometry sets with the 'start' and 'end' node of the curve. Also a 'line' set
        with all nodes of the curve.
    """

    # Check for mutually exclusive parameters
    n_given_arguments = sum(
        1
        for argument in [n_el, l_el, node_positions_of_elements]
        if argument is not None
    )
    if n_given_arguments == 0:
        # No arguments were given, use a single element per default
        n_el = 1
    elif n_given_arguments > 1:
        raise ValueError(
            'The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive'
        )

    if close_beam and end_node is not None:
        raise ValueError(
            'The arguments "close_beam" and "end_node" are mutually exclusive'
        )

    # Cases where we have equally spaced elements
    if n_el is not None or l_el is not None:
        if l_el is not None:
            # Calculate the number of elements in case a desired element length is provided
            if interval_length is None:
                raise ValueError(
                    'The parameter "l_el" requires "interval_length" to be set.'
                )
            n_el = max([1, round(interval_length / l_el)])
        elif n_el is None:
            raise ValueError("n_el should not be None at this point")

        interval_node_positions_of_elements = [
            interval[0] + i_node * (interval[1] - interval[0]) / n_el
            for i_node in range(n_el + 1)
        ]
    # A list for the element node positions was provided
    elif node_positions_of_elements is not None:
        # Check that the given positions are in ascending order and start with 1 and end with 0
        for index, value, name in zip([0, -1], [0, 1], ["First", "Last"]):
            if not _np.isclose(
                value,
                node_positions_of_elements[index],
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{name} entry of node_positions_of_elements must be {value}, got {node_positions_of_elements[index]}"
                )
        if not all(
            x < y
            for x, y in zip(node_positions_of_elements, node_positions_of_elements[1:])
        ):
            raise ValueError(
                f"The given node_positions_of_elements must be in ascending order. Got {node_positions_of_elements}"
            )
        interval_node_positions_of_elements = interval[0] + (
            interval[1] - interval[0]
        ) * _np.asarray(node_positions_of_elements)

    # We need to make sure we have the number of elements for the case a given end node
    n_el = len(interval_node_positions_of_elements) - 1

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

    def get_relative_twist(rotation_node, rotation_function):
        """Check if the rotation at a node and the one returned by the function
        match.

        If not, check if the first basis vector of the triads is the
        same. If that is the case, a simple relative twist can be
        applied to ensure that the triad field is continuous. This
        relative twist can lead to issues if the beam cross-section is
        not double symmetric.
        """

        if rotation_node == rotation_function:
            return None
        elif not _mpy.allow_beam_rotation:
            # The settings do not allow for a rotation of the beam
            raise ValueError(
                "Given nodal rotation does not match with given rotation function!"
            )
        else:
            # Evaluate the relative rotation
            # First check if the first basis vector is the same
            relative_basis_1 = rotation_node.inv() * rotation_function * [1, 0, 0]
            if _np.linalg.norm(relative_basis_1 - [1, 0, 0]) < _mpy.eps_quaternion:
                # Calculate the relative rotation
                return rotation_function.inv() * rotation_node
            else:
                raise ValueError(
                    "The tangent of the start node does not match with the given function!"
                )

    # Position and rotation at the start and end of the interval
    function_over_whole_interval = function_generator(*interval)
    relative_twist_start = None
    relative_twist_end = None

    # If a start node is given, set this as the first node for this beam.
    if start_node is not None:
        start_node = _get_single_node(start_node)
        nodes = [start_node]
        check_given_node(start_node)
        _, start_rotation = function_over_whole_interval(-1.0)
        relative_twist_start = get_relative_twist(start_node.rotation, start_rotation)

    # If an end node is given, check what behavior is wanted.
    if end_node is not None:
        end_node = _get_single_node(end_node)
        check_given_node(end_node)
        _, end_rotation = function_over_whole_interval(1.0)
        relative_twist_end = get_relative_twist(end_node.rotation, end_rotation)

    # Check if a relative twist has to be applied
    if relative_twist_start is not None and relative_twist_end is not None:
        if relative_twist_start == relative_twist_end:
            relative_twist = relative_twist_start
        else:
            raise ValueError(
                "The relative twist required for the start and end node do not match"
            )
    elif relative_twist_start is not None:
        relative_twist = relative_twist_start
    elif relative_twist_end is not None:
        relative_twist = relative_twist_end
    else:
        relative_twist = None

    # Create the beams.
    for i_el in range(len(interval_node_positions_of_elements) - 1):
        # If the beam is closed with itself, set the end node to be the
        # first node of the beam. This is done when the second element is
        # created, as the first node already exists here.
        if i_el == 1 and close_beam:
            end_node = nodes[0]

        # Get the function to create this beam element.
        function = function_generator(
            interval_node_positions_of_elements[i_el],
            interval_node_positions_of_elements[i_el + 1],
        )

        # Set the start node for the created beam.
        if start_node is not None or i_el > 0:
            first_node = nodes[-1]
        else:
            first_node = None

        # If an end node is given, set this one for the last element.
        if end_node is not None and i_el == n_el - 1:
            last_node = end_node
        else:
            last_node = None

        element = beam_object(material=material)
        elements.append(element)
        nodes.extend(
            element.create_beam(
                function,
                start_node=first_node,
                end_node=last_node,
                relative_twist=relative_twist,
            )
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
    return_set = _GeometryName()
    return_set["start"] = _GeometrySet(nodes[0])
    return_set["end"] = _GeometrySet(end_node)
    return_set["line"] = _GeometrySet(elements)

    return return_set
