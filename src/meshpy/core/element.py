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

from typing import List as _List

from meshpy.core.base_mesh_item import BaseMeshItem as _BaseMeshItem
from meshpy.core.node import Node as _Node


class Element(_BaseMeshItem):
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
    def from_fourcipp_dict(cls, nodes: _List[_Node], element: dict):
        """Create an element from a legacy string."""

        from meshpy.core.element_volume import VolumeHEX8 as _VolumeHEX8
        from meshpy.core.element_volume import VolumeHEX20 as _VolumeHEX20
        from meshpy.core.element_volume import VolumeHEX27 as _VolumeHEX27
        from meshpy.core.element_volume import VolumeTET4 as _VolumeTET4
        from meshpy.core.element_volume import VolumeTET10 as _VolumeTET10
        from meshpy.core.element_volume import VolumeWEDGE6 as _VolumeWEDGE6
        from meshpy.four_c.element_volume import SolidRigidSphere as _SolidRigidSphere

        # Depending on the number of nodes chose which solid element to return.
        match len(element["cell"]["connectivity"]):
            case 8:
                return _VolumeHEX8(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case 4:
                return _VolumeTET4(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case 10:
                return _VolumeTET10(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case 20:
                return _VolumeHEX20(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case 27:
                return _VolumeHEX27(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case 6:
                return _VolumeWEDGE6(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case 1:
                return _SolidRigidSphere(
                    nodes=nodes,
                    string_pre_nodes=element["cell"]["type"],
                    string_post_nodes=element["data"],
                )
            case _:
                raise TypeError(
                    f"Could not find a element type for {element["cell"]["type"]}, with {len(element["cell"]["connectivity"])} nodes"
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

    def dump_element_specific_section(self, yaml_dict):
        """Add information of this element to specific section (e.g. STRUCTURE
        KNOTVECTORS for NURBS elements)."""

    def get_vtk(self, vtk_writer_beam, vtk_writer_solid, **kwargs):
        """Add representation of this element to the vtk_writers for solid and
        beam."""
        raise NotImplementedError("VTK output has to be implemented in the class!")
