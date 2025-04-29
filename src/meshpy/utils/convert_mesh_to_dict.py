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
"""Utility function to convert a MeshPy mesh object into a dictionary for
easier comparison during testing and during dumping to a 4C input file."""

from typing import Dict as _Dict

from meshpy.core.conf import mpy as _mpy
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.node import Node as _Node
from meshpy.core.element import Element as _Element
from meshpy.core.material import Material as _Material

from meshpy.four_c.function import Function as _Function
from meshpy.four_c.input_file_mappings import (
    FOUR_C_INPUT_FILE_MAPPINGS as _FOUR_C_INPUT_FILE_MAPPINGS,
)

from meshpy.core.nurbs_patch import NURBSPatch as _NURBSPatch


def convert_mesh_to_dict(
    mesh: _Mesh,
    *,
    global_start_indices: dict = {_Node: 0, _Element: 0, _Material: 0, _Function: 0}
    | {
        key: 0 for key in _FOUR_C_INPUT_FILE_MAPPINGS["section_names"].keys()
    },  # TODO replace this with a generally valid dict rather than the 4C dict
    input_file_mappings: _Dict = _FOUR_C_INPUT_FILE_MAPPINGS,  # TODO replace this with a generally valid dict rather than the 4C dict
) -> _Dict:
    """Convert a MeshPy mesh object into a dictionary for easier comparison
    during testing and during dumping to a 4C input file.

    Args:
        mesh: MeshPy mesh which should be converted into a dictionary.
        global_start_indices: Dictionary containing the starting indices for
            global numbering of nodes, elements, materials, functions and
            d-topologies.
        input_file_mappings: Dictionary containing the mappings between
            MeshPy objects and input files

    Returns:
        A dictionary containing the mesh data.
    """

    dict: _Dict = {}

    # extract sets from couplings and boundary conditions to a temp container
    mesh.unlink_nodes()
    # TODO can this move to the bottom where the mesh sets are actually used?
    mesh_sets = mesh.get_unique_geometry_sets(
        geometry_set_start_indices={
            key: global_start_indices[key]
            for key in _FOUR_C_INPUT_FILE_MAPPINGS["geometry_set_names"].keys()
        }
    )

    # assign global indices to all items
    set_i_global(mesh.nodes, start_index=global_start_indices[_Node])
    set_i_global(mesh.elements, start_index=global_start_indices[_Element])
    set_i_global(mesh.materials, start_index=global_start_indices[_Material])
    set_i_global(mesh.functions, start_index=global_start_indices[_Function])

    # add materials
    add_items_to_dict(
        dict, input_file_mappings["section_names"][_Material], mesh.materials
    )

    # add functions
    for function in mesh.functions:
        dict[
            f"{input_file_mappings["section_names"][_Function]}{function.i_global}"
        ] = function.dump_data()

    # If there are couplings in the mesh, set the link between the nodes
    # and elements, so the couplings can decide which DOFs they couple,
    # depending on the type of the connected beam element.
    # ! TODO correct this
    # if any(
    #     (k, _mpy.geo.point) in mesh.boundary_conditions
    #     for k in [_mpy.bc.point_coupling, _mpy.bc.point_coupling_penalty]
    # ):
    #     mesh.set_node_links()

    # add boundary conditions.
    for (bc_key, geom_key), bc_list in mesh.boundary_conditions.items():
        if len(bc_list) > 0:
            section_name = (
                bc_key
                if isinstance(bc_key, str)
                else _FOUR_C_INPUT_FILE_MAPPINGS["boundary_condition_names"][
                    bc_key, geom_key
                ]
            )
            add_items_to_dict(dict, section_name, bc_list)

    # add additional element sections, e.g., for NURBS knot vectors
    for element in mesh.elements:
        element.dump_element_specific_section(dict)

    # add the geometry sets
    # TODO update this mechanism to also use the `add_items_to_dict` function
    for geom_key, geom_sets in mesh_sets.items():
        if len(geom_sets) > 0:
            for item in geom_sets:
                for node in item.dump_to_list():
                    dict.setdefault(
                        _FOUR_C_INPUT_FILE_MAPPINGS["geometry_set_names"][geom_key],
                        [],
                    ).append(node)

    # Add the nodes and elements.
    add_items_to_dict(dict, input_file_mappings["section_names"][_Node], mesh.nodes)
    add_items_to_dict(
        dict, input_file_mappings["section_names"][_Element], mesh.elements
    )

    return dict


def add_items_to_dict(dict, section_name, items):
    """Add a list of items to the dictionary under the given section name.

    Args:
        dict: The dictionary to which the items should be added.
        section_name: The name of the section under which the items should be added.
        items: The list of items to be added.
    """

    for item in items:
        dict.setdefault(section_name, []).append(item.dump_data())


def set_i_global(data: list, start_index: int) -> None:
    """Set i_global for each item of the provided data.

    Args:
        data: List of items to set i_global for.
        start_index: Starting index for i_global.
    """
    # Check if each item in data is unique
    if len(data) != len(set(data)):
        raise ValueError("Elements in data are not unique!")

    # regular numbering
    if not any(isinstance(item, _NURBSPatch) for item in data):
        for i, item in enumerate(data):
            item.i_global = i + 1 + start_index  # TODO make i_global index-0 based

    # special treatment if one item in data is a NURBSPatch
    else:
        i_nurbs_patch = 0

        for item in data:
            # As a NURBS patch can be defined with more elements, an offset is applied to the
            # rest of the items
            item.i_global = start_index + 1  # TODO make i_global index-0 based
            if isinstance(item, _NURBSPatch):
                item.n_nurbs_patch = i_nurbs_patch + 1
                offset = item.get_number_elements()
                start_index += offset
                i_nurbs_patch += 1
            else:
                start_index += 1
