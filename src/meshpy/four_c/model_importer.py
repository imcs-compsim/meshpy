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
"""This module contains functions to load and parse existing 4C input files."""

from pathlib import Path as _Path
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Union as _Union

import yaml as _yaml

import meshpy.core.conf as _conf
from meshpy.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from meshpy.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from meshpy.core.conf import mpy as _mpy
from meshpy.core.coupling import Coupling as _Coupling
from meshpy.core.element_volume import VolumeHEX8 as _VolumeHEX8
from meshpy.core.element_volume import VolumeHEX20 as _VolumeHEX20
from meshpy.core.element_volume import VolumeHEX27 as _VolumeHEX27
from meshpy.core.element_volume import VolumeTET4 as _VolumeTET4
from meshpy.core.element_volume import VolumeTET10 as _VolumeTET10
from meshpy.core.element_volume import VolumeWEDGE6 as _VolumeWEDGE6
from meshpy.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.node import Node as _Node
from meshpy.four_c.element_volume import SolidRigidSphere as _SolidRigidSphere
from meshpy.four_c.input_file import InputFile as _InputFile
from meshpy.four_c.input_file import (
    get_geometry_set_indices_from_section as _get_geometry_set_indices_from_section,
)
from meshpy.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from meshpy.utils.environment import fourcipp_is_available as _fourcipp_is_available


def import_four_c_model(
    input_file_path: _Path, convert_input_to_mesh: bool = False
) -> _Tuple[_InputFile, _Mesh]:
    """Import an existing 4C input file and optionally convert it into a MeshPy
    mesh.

    Args:
        input_file_path: A file path to an existing 4C input file that will be
            imported.
        convert_input_to_mesh: If True, the input file will be converted to a
            MeshPy mesh.

    Returns:
        A tuple with the input file and the mesh. If convert_input_to_mesh is
        False, the mesh will be empty. Note that the input sections which are
        converted to a MeshPy mesh are removed from the input file object.
    """

    if _fourcipp_is_available():
        raise ValueError("Use fourcipp to parse the yaml file.")

    with open(input_file_path) as stream:
        input_file = _InputFile()
        input_file.sections = _yaml.safe_load(stream)

    if convert_input_to_mesh:
        return _extract_mesh_sections(input_file)
    else:
        return input_file, _Mesh()


def _element_from_dict(nodes: _List[_Node], input_line: str):
    """TODO: Update this doc string once we don't use the legacy string any more.
    Create an element from a legacy string."""

    if _fourcipp_is_available():
        raise ValueError(
            "Port this functionality to create the element from the dict "
            "representing the element, not the legacy string."
            "TODO: pass the nodes array here, so we can directly link to the nodes"
            "TODO: The whole string_pre_nodes and string_post_nodes is obsolete once"
            " we move on from legacy string"
        )

    # Split up input line and get pre node string.
    line_split = input_line.split()
    string_pre_nodes = " ".join(line_split[1:3])

    # Get a list of the element nodes.
    # This is only here because we need the pre and post strings - can be
    # removed when moving on from the legacy format.
    dummy = []
    for i, item in enumerate(line_split[3:]):
        if item.isdigit():
            dummy.append(int(item) - 1)
        else:
            break
    else:
        raise ValueError(
            f'The input line:\n"{input_line}"\ncould not be converted to a solid element!'
        )

    # Get the post node string
    string_post_nodes = " ".join(line_split[3 + i :])

    # Depending on the number of nodes chose which solid element to return.
    n_nodes = len(nodes)
    element_type = {
        8: _VolumeHEX8,
        20: _VolumeHEX20,
        27: _VolumeHEX27,
        4: _VolumeTET4,
        10: _VolumeTET10,
        6: _VolumeWEDGE6,
        1: _SolidRigidSphere,
    }
    if n_nodes not in element_type:
        raise TypeError(
            f"Could not find a element type for {string_pre_nodes}, with {n_nodes} nodes"
        )
    return element_type[n_nodes](
        nodes=nodes,
        string_pre_nodes=string_pre_nodes,
        string_post_nodes=string_post_nodes,
    )


def _boundary_condition_from_dict(
    geometry_set: _GeometrySetNodes,
    bc_key: _Union[_conf.BoundaryCondition, str],
    data: _Dict,
) -> _BoundaryConditionBase:
    """This function acts as a factory and creates the correct boundary
    condition object from a dictionary parsed from an input file."""

    del data["E"]

    if bc_key in (
        _mpy.bc.dirichlet,
        _mpy.bc.neumann,
        _mpy.bc.locsys,
        _mpy.bc.beam_to_solid_surface_meshtying,
        _mpy.bc.beam_to_solid_surface_contact,
        _mpy.bc.beam_to_solid_volume_meshtying,
    ) or isinstance(bc_key, str):
        return _BoundaryCondition(geometry_set, data, bc_type=bc_key)
    elif bc_key is _mpy.bc.point_coupling:
        return _Coupling(geometry_set, bc_key, data, check_overlapping_nodes=False)
    else:
        raise ValueError("Got unexpected boundary condition!")


def _get_yaml_geometry_sets(
    nodes: _List[_Node], geometry_key: _conf.Geometry, section_list: _List
) -> _Dict[int, _GeometrySetNodes]:
    """Add sets of points, lines, surfaces or volumes to the object."""

    # Create the individual geometry sets
    geometry_set_dict = _get_geometry_set_indices_from_section(section_list)
    geometry_sets_in_this_section = {}
    for geometry_set_id, node_ids in geometry_set_dict.items():
        geometry_sets_in_this_section[geometry_set_id] = _GeometrySetNodes(
            geometry_key, nodes=[nodes[node_id] for node_id in node_ids]
        )
    return geometry_sets_in_this_section


def _extract_mesh_sections(input_file: _InputFile) -> _Tuple[_InputFile, _Mesh]:
    """Convert an existing input file to a MeshPy mesh with mesh items, e.g.,
    nodes, elements, element sets, node sets, boundary conditions, materials.

    Args:
        input_file: The input file object that contains the sections to be
            converted to a MeshPy mesh.
    Returns:
        A tuple with the input file and the mesh. The input file will be
            modified to remove the sections that have been converted to a
            MeshPy mesh.
    """

    def _get_section_items(section_name):
        """Return the items in a given section.

        Since we will add the created MeshPy objects to the mesh, we
        delete them from the plain data storage to avoid having
        duplicate entries.
        """
        if section_name in input_file.sections:
            return_list = input_file.sections[section_name]
            input_file.sections[section_name] = []
        else:
            return_list = []
        return return_list

    # Go through all sections that have to be converted to full MeshPy objects
    mesh = _Mesh()

    # Add nodes
    for item in _get_section_items("NODE COORDS"):
        mesh.nodes.append(_Node.from_legacy_string(item))

    # Add elements
    for item in _get_section_items("STRUCTURE ELEMENTS"):
        if _fourcipp_is_available():
            raise ValueError("Port this functionality to not use the legacy string.")

        # Get a list containing the element nodes.
        element_nodes = []
        for split_item in item.split()[3:]:
            if split_item.isdigit():
                node_id = int(split_item) - 1
                element_nodes.append(mesh.nodes[node_id])
            else:
                break
        else:
            raise ValueError(
                f'The input line:\n"{item}"\ncould not be converted to a element!'
            )

        mesh.elements.append(_element_from_dict(element_nodes, item))

    # Add geometry sets
    geometry_sets_in_sections: dict[str, dict[int, _GeometrySetNodes]] = {
        key: {} for key in _mpy.geo
    }
    for section_name in input_file.sections.keys():
        if section_name.endswith("TOPOLOGY"):
            section_items = _get_section_items(section_name)
            if len(section_items) > 0:
                # Get the geometry key for this set
                for key, value in _INPUT_FILE_MAPPINGS["geometry_sets"].items():
                    if value == section_name:
                        geometry_key = key
                        break
                else:
                    raise ValueError(f"Could not find the set {section_name}")
                geometry_sets_in_section = _get_yaml_geometry_sets(
                    mesh.nodes, geometry_key, section_items
                )
                geometry_sets_in_sections[geometry_key] = geometry_sets_in_section
                mesh.geometry_sets[geometry_key] = list(
                    geometry_sets_in_section.values()
                )

    # Add boundary conditions
    for (
        bc_key,
        geometry_key,
    ), section_name in _INPUT_FILE_MAPPINGS["boundary_conditions"].items():
        for item in _get_section_items(section_name):
            geometry_set_id = item["E"]
            geometry_set = geometry_sets_in_sections[geometry_key][geometry_set_id]
            mesh.boundary_conditions.append(
                (bc_key, geometry_key),
                _boundary_condition_from_dict(geometry_set, bc_key, item),
            )

    return input_file, mesh
