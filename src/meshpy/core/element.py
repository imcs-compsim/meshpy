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
"""This module implements the class that represents one element in the Mesh."""

from meshpy.core.base_mesh_item import BaseMeshItemFull


class Element(BaseMeshItemFull):
    """A base class for an FEM element in the mesh."""

    def __init__(self, nodes=None, material=None, **kwargs):
        super().__init__(data=None, **kwargs)

        # List of nodes that are connected to the element.
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

        # Material of this element.
        self.material = material

        # VTK cell data for this element.
        self.vtk_cell_data = {}

    @classmethod
    def from_dat(cls, input_line):
        """Check the string to decide which element to use.

        Nodes are linked with the 0 based index and will be connected to
        the Node objects after the whole input file is parsed.
        """

        # Import solid element classes for creation of the element.
        from meshpy.core.element_volume import (
            VolumeHEX8,
            VolumeHEX20,
            VolumeHEX27,
            VolumeTET4,
            VolumeTET10,
        )
        from meshpy.four_c.element_volume import SolidRigidSphere

        # Split up input line and get pre node string.
        line_split = input_line[0].split()
        dat_pre_nodes = " ".join(line_split[1:3])

        # Get a list of the element nodes.
        element_nodes = []
        for i, item in enumerate(line_split[3:]):
            if item.isdigit():
                element_nodes.append(int(item) - 1)
            else:
                break
        else:
            raise ValueError(
                f'The input line:\n"{input_line}"\ncould not be converted to a solid element!'
            )

        # Get the post node string
        dat_post_nodes = " ".join(line_split[3 + i :])

        # Depending on the number of nodes chose which solid element to return.
        n_nodes = len(element_nodes)
        match n_nodes:
            case 8:
                return VolumeHEX8(
                    nodes=element_nodes,
                    dat_pre_nodes=dat_pre_nodes,
                    dat_post_nodes=dat_post_nodes,
                    comments=input_line[1],
                )
            case 4:
                return VolumeTET4(
                    nodes=element_nodes,
                    dat_pre_nodes=dat_pre_nodes,
                    dat_post_nodes=dat_post_nodes,
                    comments=input_line[1],
                )
            case 10:
                return VolumeTET10(
                    nodes=element_nodes,
                    dat_pre_nodes=dat_pre_nodes,
                    dat_post_nodes=dat_post_nodes,
                    comments=input_line[1],
                )
            case 20:
                return VolumeHEX20(
                    nodes=element_nodes,
                    dat_pre_nodes=dat_pre_nodes,
                    dat_post_nodes=dat_post_nodes,
                    comments=input_line[1],
                )
            case 27:
                return VolumeHEX27(
                    nodes=element_nodes,
                    dat_pre_nodes=dat_pre_nodes,
                    dat_post_nodes=dat_post_nodes,
                    comments=input_line[1],
                )
            case 1:
                return SolidRigidSphere(
                    nodes=element_nodes,
                    dat_pre_nodes=dat_pre_nodes,
                    dat_post_nodes=dat_post_nodes,
                    comments=input_line[1],
                )
            case _:
                raise TypeError(
                    f"Could not find a element type for {dat_pre_nodes}, with {n_nodes} nodes"
                )

    def flip(self):
        """Reverse the nodes of this element.

        This is usually used when reflected.
        """
        raise NotImplementedError(
            f"The flip method is not implemented for {self.__class__}"
        )

    def replace_node(self, old_node, new_node):
        """Replace old_node with new_node."""

        # Look for old_node and replace it. If it is not found, throw error.
        for i, node in enumerate(self.nodes):
            if node == old_node:
                self.nodes[i] = new_node
                break
        else:
            raise ValueError(
                "The node that should be replaced is not in the current element"
            )

    def add_element_specific_section(self, sections):
        """Add element specific section (e.g. STRUCTURE KNOTVECTORS for NURBS
        elements) to the sections dictionary."""

    def get_vtk(self, vtk_writer_beam, vtk_writer_solid, **kwargs):
        """Add representation of this element to the vtk_writers for solid and
        beam."""
        raise NotImplementedError("VTK output has to be implemented in the class!")
