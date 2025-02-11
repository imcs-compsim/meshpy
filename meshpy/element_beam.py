# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This module implements beam elements for the mesh."""

import warnings
from typing import Any, Optional

import numpy as np
import vtk

from .conf import mpy
from .element import Element
from .material import (
    MaterialEulerBernoulli,
    MaterialKirchhoff,
    MaterialReissner,
    MaterialReissnerElastoplastic,
)
from .node import NodeCosserat
from .vtk_writer import add_point_data_node_sets


class Beam(Element):
    """A base class for a beam element."""

    # An array that defines the parameter positions of the element nodes,
    # in ascending order.
    nodes_create: Any = []

    # A list of valid material types for this element.
    valid_material: Any = []

    # Coupling strings.
    coupling_fix_string: Optional[str] = None
    coupling_joint_string: Optional[str] = None

    def __init__(self, material=None, nodes=None):
        super().__init__(nodes=nodes, material=material)

    def create_beam(
        self, beam_function, *, start_node=None, end_node=None, relative_twist=None
    ):
        """Create the nodes for this beam element. The function returns a list
        with the created nodes.

        In the case of start_node and end_node, it is checked, that the
        function and the node have the same coordinates and rotations.

        Args
        ----
        beam_function: function(xi)
            Returns the position and rotation of the beam along the local
            coordinate xi.
        start_node: Node
            If this argument is given, this is the node of the beam at xi=-1.
        end_node: Node
            If this argument is given, this is the node of the beam at xi=1.
        relative_twist: Rotation
            Apply this relative rotation to all created nodes. This can be used to
            "twist" the created beam to match the rotation of given nodes.
        """

        if len(self.nodes) > 0:
            raise ValueError("The beam should not have any local nodes yet!")

        def check_node(node, pos, rot, name):
            """Check if the given node matches with the position and
            rotation."""

            if np.linalg.norm(pos - node.coordinates) > mpy.eps_pos:
                raise ValueError(
                    f"{name} position does not match with function! Got {pos} from function but "
                    + f"given node value is {node.coordinates}"
                )
            if not node.rotation == rot:
                raise ValueError(f"{name} rotation does not match with function!")

        # Flags if nodes are given
        has_start_node = start_node is not None
        has_end_node = end_node is not None

        # Loop over local nodes.
        for i, xi in enumerate(self.nodes_create):
            # Get the position and rotation at xi
            pos, rot = beam_function(xi)
            if relative_twist is not None:
                rot = rot * relative_twist

            # Check if the position and rotation match existing nodes
            if i == 0 and has_start_node:
                check_node(start_node, pos, rot, "start_node")
                self.nodes = [start_node]
            elif (i == len(self.nodes_create) - 1) and has_end_node:
                check_node(end_node, pos, rot, "end_node")

            # Create the node
            if (i > 0 or not has_start_node) and (
                i < len(self.nodes_create) - 1 or not has_end_node
            ):
                is_middle_node = 0 < i < len(self.nodes_create) - 1
                self.nodes.append(NodeCosserat(pos, rot, is_middle_node=is_middle_node))

        # Get a list with the created nodes.
        if has_start_node:
            created_nodes = self.nodes[1:]
        else:
            created_nodes = self.nodes.copy()

        # Add the end node to the beam.
        if has_end_node:
            self.nodes.append(end_node)

        # Return the created nodes.
        return created_nodes

    @classmethod
    def get_coupling_string(cls, coupling_dof_type):
        """Return the string to couple this beam to another beam."""

        match coupling_dof_type:
            case mpy.coupling_dof.joint:
                if cls.coupling_joint_string is None:
                    raise ValueError(f"Joint coupling is not implemented for {cls}")
                return cls.coupling_joint_string
            case mpy.coupling_dof.fix:
                if cls.coupling_fix_string is None:
                    raise ValueError("Fix coupling is not implemented for {cls}")
                return cls.coupling_fix_string
            case _:
                raise ValueError(
                    f'Coupling_dof_type "{coupling_dof_type}" is not implemented!'
                )

    def flip(self):
        """Reverse the nodes of this element.

        This is usually used when reflected.
        """
        self.nodes = [self.nodes[-1 - i] for i in range(len(self.nodes))]

    def _check_material(self):
        """Check if the linked material is valid for this type of beam
        element."""
        for material_type in self.valid_material:
            if isinstance(self.material, material_type):
                break
        else:
            raise TypeError(
                f"Beam of type {type(self)} can not have a material of type {type(self.material)}!"
            )

    def get_vtk(
        self,
        vtk_writer_beam,
        vtk_writer_solid,
        *,
        beam_centerline_visualization_segments=1,
        **kwargs,
    ):
        """Add the representation of this element to the VTK writer as a poly
        line.

        Args
        ----
        vtk_writer_beam:
            VTK writer for the beams.
        vtk_writer_solid:
            VTK writer for solid elements, not used in this method.
        beam_centerline_visualization_segments: int
            Number of segments to be used for visualization of beam centerline between successive
            nodes. Default is 1, which means a straight line is drawn between the beam nodes. For
            Values greater than 1, a Hermite interpolation of the centerline is assumed for
            visualization purposes.
        """

        n_nodes = len(self.nodes)
        n_segments = n_nodes - 1
        n_additional_points_per_segment = beam_centerline_visualization_segments - 1
        # Number of points (in addition to the nodes) to be used for output
        n_additional_points = n_segments * n_additional_points_per_segment
        n_points = n_nodes + n_additional_points

        # Dictionary with cell data.
        cell_data = self.vtk_cell_data.copy()
        cell_data["cross_section_radius"] = self.material.radius

        # Dictionary with point data.
        point_data = {}
        point_data["node_value"] = np.zeros(n_points)
        point_data["base_vector_1"] = np.zeros((n_points, 3))
        point_data["base_vector_2"] = np.zeros((n_points, 3))
        point_data["base_vector_3"] = np.zeros((n_points, 3))

        coordinates = np.zeros((n_points, 3))
        nodal_rotation_matrices = [
            node.rotation.get_rotation_matrix() for node in self.nodes
        ]

        for i_node, (node, rotation_matrix) in enumerate(
            zip(self.nodes, nodal_rotation_matrices)
        ):
            coordinates[i_node, :] = node.coordinates
            if node.is_middle_node:
                point_data["node_value"][i_node] = 0.5
            else:
                point_data["node_value"][i_node] = 1.0

            point_data["base_vector_1"][i_node] = rotation_matrix[:, 0]
            point_data["base_vector_2"][i_node] = rotation_matrix[:, 1]
            point_data["base_vector_3"][i_node] = rotation_matrix[:, 2]

        # We can output the element partner index, which is a debug quantity to help find elements
        # with matching middle nodes. This is usually an indicator for an issue with the mesh.
        # TODO: Check if we really need this partner index output any more
        element_partner_indices = list(
            [
                node.element_partner_index
                for node in self.nodes
                if node.element_partner_index is not None
            ]
        )
        if len(element_partner_indices) > 1:
            raise ValueError(
                "More than one element partner indices are currently not supported in the output"
                "functionality"
            )
        elif len(element_partner_indices) == 1:
            cell_data["partner_index"] = element_partner_indices[0] + 1

        # Check if we have everything we need to write output or if we need to calculate additional
        # points for a smooth beam visualization.
        if beam_centerline_visualization_segments == 1:
            point_connectivity = np.arange(n_nodes)
        else:
            # We need the centerline shape function matrices, so calculate them once and use for
            # all segments that we need. Drop the first and last value, since they represent the
            # nodes which we have already added above.
            xi = np.linspace(-1, 1, beam_centerline_visualization_segments + 1)[1:-1]
            hermite_shape_functions_pos = np.array(
                [
                    0.25 * (2.0 + xi) * (1.0 - xi) ** 2,
                    0.25 * (2.0 - xi) * (1.0 + xi) ** 2,
                ]
            ).transpose()
            hermite_shape_functions_tan = np.array(
                [
                    0.125 * (1.0 + xi) * (1.0 - xi) ** 2,
                    -0.125 * (1.0 - xi) * (1.0 + xi) ** 2,
                ]
            ).transpose()

            point_connectivity = np.zeros(n_points, dtype=int)

            for i_segment in range(n_segments):
                positions = np.array(
                    [
                        self.nodes[i_node].coordinates
                        for i_node in [i_segment, i_segment + 1]
                    ]
                )
                tangents = np.array(
                    [
                        nodal_rotation_matrices[i_node][:, 0]
                        for i_node in [i_segment, i_segment + 1]
                    ]
                )
                length_factor = np.linalg.norm(positions[1] - positions[0])
                interpolated_coordinates = np.dot(
                    hermite_shape_functions_pos, positions
                ) + length_factor * np.dot(hermite_shape_functions_tan, tangents)

                index_first_point = (
                    n_nodes + i_segment * n_additional_points_per_segment
                )
                index_last_point = (
                    n_nodes + (i_segment + 1) * n_additional_points_per_segment
                )

                coordinates[index_first_point:index_last_point] = (
                    interpolated_coordinates
                )
                point_connectivity[
                    i_segment * beam_centerline_visualization_segments
                ] = i_segment
                point_connectivity[
                    (i_segment + 1) * beam_centerline_visualization_segments
                ] = i_segment + 1
                point_connectivity[
                    i_segment * beam_centerline_visualization_segments + 1 : (
                        i_segment + 1
                    )
                    * beam_centerline_visualization_segments
                ] = np.arange(index_first_point, index_last_point)

        # Get the point data sets and add everything to the output file.
        add_point_data_node_sets(
            point_data, self.nodes, extra_points=n_additional_points
        )
        indices = vtk_writer_beam.add_points(coordinates, point_data=point_data)
        vtk_writer_beam.add_cell(
            vtk.vtkPolyLine, indices[point_connectivity], cell_data=cell_data
        )


class Beam3rHerm2Line3(Beam):
    """Represents a BEAM3R HERM2LINE3 element."""

    nodes_create = [-1, 0, 1]
    beam_type = mpy.beam.reissner
    valid_material = [MaterialReissner, MaterialReissnerElastoplastic]

    coupling_fix_string = "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0"
    coupling_joint_string = "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0"

    def _get_dat(self):
        """Return the line for the input file."""

        string_nodes = ""
        string_triads = ""
        for i in [0, 2, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        return (
            f"{self.i_global} BEAM3R HERM2LINE3 {string_nodes}MAT {self.material.i_global} "
            f"TRIADS{string_triads}"
        )


class Beam3rLine2Line2(Beam):
    """Represents a Reissner beam with linear shapefunctions in the rotations
    as well as the displacements."""

    nodes_create = [-1, 1]
    beam_type = mpy.beam.reissner
    valid_material = [MaterialReissner]

    coupling_fix_string = "NUMDOF 6 ONOFF 1 1 1 1 1 1"
    coupling_joint_string = "NUMDOF 6 ONOFF 1 1 1 0 0 0"

    def _get_dat(self):
        """Return the line for the input file."""

        string_nodes = ""
        string_triads = ""
        for i in [0, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        return (
            f"{self.i_global} BEAM3R LINE2 {string_nodes}MAT {self.material.i_global} "
            f"TRIADS{string_triads}"
        )


class Beam3kClass(Beam):
    """Represents a Kirchhoff beam element."""

    nodes_create = [-1, 0, 1]
    beam_type = mpy.beam.kirchhoff
    valid_material = [MaterialKirchhoff]

    coupling_fix_string = "NUMDOF 7 ONOFF 1 1 1 1 1 1 0"
    coupling_joint_string = "NUMDOF 7 ONOFF 1 1 1 0 0 0 0"

    def __init__(self, *, weak=True, rotvec=True, is_fad=True, **kwargs):
        Beam.__init__(self, **kwargs)

        # Set the parameters for this beam.
        self.weak = weak
        self.rotvec = rotvec
        self.is_fad = is_fad

        # Show warning when not using rotvec.
        if not rotvec:
            warnings.warn(
                "Use rotvec=False with caution, especially when applying the boundary conditions "
                "and couplings."
            )

    def _get_dat(self):
        """Return the line for the input file."""

        string_nodes = ""
        string_triads = ""
        for i in [0, 2, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        string_dat = ("{} BEAM3K LINE3 {} WK {} ROTVEC {} MAT {} TRIADS{}{}").format(
            self.i_global,
            string_nodes,
            "1" if self.weak else "0",
            "1" if self.rotvec else "0",
            self.material.i_global,
            string_triads,
            " FAD" if self.is_fad else "",
        )

        return string_dat


def Beam3k(**kwargs_class):
    """This factory returns a function that creates a new Beam3kClass object
    with certain attributes defined.

    The returned function behaves like a call to the object.
    """

    def create_class(**kwargs):
        """The function that will be returned.

        This function should behave like the call to the __init__
        function of the class.
        """
        return Beam3kClass(**kwargs_class, **kwargs)

    return create_class


class Beam3eb(Beam):
    """Represents a Euler Bernoulli beam element."""

    nodes_create = [-1, 1]
    beam_type = mpy.beam.euler_bernoulli
    valid_material = [MaterialEulerBernoulli]

    def _get_dat(self):
        """Return the line for the input file."""

        # The two rotations must be the same and the x1 vector must point from
        # the start point to the end point.
        if not self.nodes[0].rotation == self.nodes[1].rotation:
            raise ValueError(
                "The two nodal rotations in Euler Bernoulli beams must be the same, i.e. the beam "
                "has to be straight!"
            )
        direction = self.nodes[1].coordinates - self.nodes[0].coordinates
        t1 = self.nodes[0].rotation * [1, 0, 0]
        if np.linalg.norm(direction / np.linalg.norm(direction) - t1) >= mpy.eps_pos:
            raise ValueError(
                "The rotations do not match the direction of the Euler Bernoulli beam!"
            )

        string_nodes = ""
        string_triads = ""
        for i in [0, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        return (
            f"{self.i_global} BEAM3EB LINE2 {string_nodes}MAT {self.material.i_global}"
        )
