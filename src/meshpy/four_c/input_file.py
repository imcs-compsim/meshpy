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
"""This module defines the classes that are used to create an input file for
4C."""

import os as _os
import sys as _sys
from datetime import datetime as _datetime
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List

import yaml as _yaml

from meshpy.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from meshpy.core.conf import mpy as _mpy
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.nurbs_patch import NURBSPatch as _NURBSPatch
from meshpy.four_c.input_file_mappings import (
    boundary_condition_names as _boundary_condition_names,
)
from meshpy.four_c.input_file_mappings import geometry_set_names as _geometry_set_names
from meshpy.four_c.yaml_dumper import MeshPyDumper as _MeshPyDumper
from meshpy.utils.environment import cubitpy_is_available as _cubitpy_is_available
from meshpy.utils.environment import fourcipp_is_available as _fourcipp_is_available
from meshpy.utils.environment import get_git_data as _get_git_data

if _cubitpy_is_available():
    import cubitpy as _cubitpy


def get_geometry_set_indices_from_section(
    section_list: _List, *, append_node_ids: bool = True
) -> _Dict:
    """Return a dictionary with the geometry set ID as keys and the node IDs as
    values.

    Args:
        section_list: A list with the legacy strings for the geometry pair
        append_node_ids: If the node IDs shall be appended, or only the
            dict with the keys should be returned.
    """

    if _fourcipp_is_available():
        raise ValueError("Port this functionality to not use the legacy string.")

    geometry_set_dict: _Dict[int, _List[int]] = {}
    for line in section_list:
        id_geometry_set = int(line.split()[-1])
        index_node = int(line.split()[1]) - 1
        if id_geometry_set not in geometry_set_dict:
            geometry_set_dict[id_geometry_set] = []
        if append_node_ids:
            geometry_set_dict[id_geometry_set].append(index_node)

    return geometry_set_dict


class InputFile:
    """An item that represents a complete 4C input file."""

    def __init__(self):
        """Initialize the input file."""

        # Everything that is not a full MeshPy object is stored here, e.g., parameters
        # and imported nodes/elements/materials/...
        self.sections: _Dict[str, _Any] = dict()

        # Contents of NOX xml file.
        self.nox_xml_contents = ""

    def add(self, object_to_add, **kwargs):
        """Add a mesh or a dictionary to the input file.

        Args:
            object: The object to be added. This can be a mesh or a dictionary.
            **kwargs: Additional arguments to be passed to the add method.
        """

        # TODO rework this once fourcipp is available to call
        # the super method directly

        if isinstance(object_to_add, _Mesh):
            self.add_mesh_to_input_file(mesh=object_to_add, **kwargs)

        elif isinstance(object_to_add, dict):
            self.add_section(object_to_add, **kwargs)

        else:
            raise TypeError(
                f"Cannot add object of type {type(object_to_add)} to the input file."
                " Only MeshPy meshes and dictionaries are supported."
            )

    def add_section(self, section, *, option_overwrite=False):
        """Add a section to the object.

        If the section name already exists, it is added to that section.
        """

        for section_name, section_value in section.items():
            if section_name in self.sections:
                section_data = self.sections[section_name]
                if isinstance(section_data, list):
                    section_data.extend(section_value)
                else:
                    for option_key, option_value in section_value.items():
                        if option_key in self.sections[section_name]:
                            if (
                                not self.sections[section_name][option_key]
                                == option_value
                            ):
                                if not option_overwrite:
                                    raise KeyError(
                                        f"Key {option_key} with the value {self.sections[section_name][option_key]} already set. You tried to set it to {option_value}"
                                    )
                        self.sections[section_name][option_key] = option_value
            else:
                self.sections[section_name] = section_value

    def delete_section(self, section_name):
        """Delete a section from the dictionary self.sections."""
        if section_name in self.sections.keys():
            del self.sections[section_name]
        else:
            raise Warning(f"Section {section_name} does not exist!")

    def write_input_file(
        self,
        file_path: _Path,
        *,
        nox_xml_file: bool | str = False,
        add_header_default: bool = True,
        add_header_information: bool = True,
        add_footer_application_script: bool = True,
    ):
        """Write the input file to disk.

        Args:
            file_path:
                Path to the input file that should be created.
            nox_xml_file:
                If this is a string, the xml file will be created with this
                name. If this is False, no xml file will be created. If this
                is True, the xml file will be created with the name of the
                input file with the extension ".xml".
            add_header_default:
                Prepend the default MeshPy header comment to the input file.
            add_header_information:
                If the information header should be exported to the input file
                Contains creation date, git details of MeshPy, CubitPy and
                original application which created the input file if available.
            add_footer_application_script:
                Append the application script which creates the input files as a
                comment at the end of the input file.
        """

        if self.nox_xml_contents:
            if nox_xml_file is False:
                xml_file_name = "NOT_DEFINED"
            elif nox_xml_file is True:
                xml_file_name = _os.path.splitext(file_path)[0] + ".xml"
            elif isinstance(nox_xml_file, str):
                xml_file_name = nox_xml_file
            else:
                raise TypeError("nox_xml_file must be a string or a boolean value.")

            self.sections["STRUCT NOX/Status Test"] = {"XML File": xml_file_name}

            # Write the xml file to the disc.
            with open(
                _os.path.join(_os.path.dirname(file_path), xml_file_name), "w"
            ) as xml_file:
                xml_file.write(self.nox_xml_contents)

        # Add information header to the input file
        if add_header_information:
            self.sections["TITLE"] = self._get_header()

        with open(file_path, "w") as input_file:
            # write MeshPy header
            if add_header_default:
                input_file.writelines(
                    "# " + line + "\n" for line in _mpy.input_file_meshpy_header
                )

            _yaml.dump(
                self.sections,
                input_file,
                Dumper=_MeshPyDumper,
                width=float("inf"),
            )

            # Add the application script to the input file.
            if add_footer_application_script:
                application_path = _Path(_sys.argv[0]).resolve()
                application_script_lines = self._get_application_script(
                    application_path
                )
                input_file.writelines(application_script_lines)

    def add_mesh_to_input_file(self, mesh: _Mesh):
        """Add a mesh to the input file.

        Args:
            mesh: The mesh to be added to the input file.
        """

        # Perform some checks on the mesh.
        if _mpy.check_overlapping_elements:
            mesh.check_overlapping_elements()

        def _get_global_start_geometry_set(dictionary):
            """Get the indices for the first "real" MeshPy geometry sets."""

            start_indices_geometry_set = {}
            for geometry_type, section_name in _geometry_set_names.items():
                max_geometry_set_id = 0
                if section_name in dictionary:
                    section_list = dictionary[section_name]
                    if len(section_list) > 0:
                        geometry_set_dict = get_geometry_set_indices_from_section(
                            section_list, append_node_ids=False
                        )
                        max_geometry_set_id = max(geometry_set_dict.keys())
                start_indices_geometry_set[geometry_type] = max_geometry_set_id
            return start_indices_geometry_set

        def _get_global_start_node(dictionary):
            """Get the index for the first "real" MeshPy node."""

            if _fourcipp_is_available():
                raise ValueError(
                    "Port this functionality to not use the legacy format any more"
                    "TODO: Check if we really want this - should we just assume that the"
                    "imported nodes are in order and without any 'missing' nodes?"
                )

            section_name = "NODE COORDS"
            if section_name in dictionary:
                return len(dictionary[section_name])
            else:
                return 0

        def _get_global_start_element(dictionary):
            """Get the index for the first "real" MeshPy element."""

            if _fourcipp_is_available():
                raise ValueError(
                    "Port this functionality to not use the legacy format any more"
                    "TODO: Check if we really want this - should we just assume that the"
                    "imported elements are in order and without any 'missing' elements?"
                )

            start_index = 0
            section_names = ["FLUID ELEMENTS", "STRUCTURE ELEMENTS"]
            for section_name in section_names:
                if section_name in dictionary:
                    start_index += len(dictionary[section_name])
            return start_index

        def _get_global_start_material(dictionary):
            """Get the index for the first "real" MeshPy material.

            We have to account for materials imported from yaml files
            that have arbitrary numbering.
            """

            # Get the maximum material index in materials imported from a yaml file
            max_material_id = 0
            section_name = "MATERIALS"
            if section_name in dictionary:
                for material in dictionary[section_name]:
                    max_material_id = max(max_material_id, material["MAT"])
            return max_material_id

        def _get_global_start_function(dictionary):
            """Get the index for the first "real" MeshPy function."""

            max_function_id = 0
            for section_name in dictionary.keys():
                if section_name.startswith("FUNCT"):
                    max_function_id = max(
                        max_function_id, int(section_name.split("FUNCT")[-1])
                    )
            return max_function_id

        def _set_i_global(data_list, *, start_index=0):
            """Set i_global in every item of data_list."""

            # A check is performed that every entry in data_list is unique.
            if len(data_list) != len(set(data_list)):
                raise ValueError("Elements in data_list are not unique!")

            # Set the values for i_global.
            for i, item in enumerate(data_list):
                # TODO make i_global index-0 based
                item.i_global = i + 1 + start_index

        def _set_i_global_elements(element_list, *, start_index=0):
            """Set i_global in every item of element_list."""

            # A check is performed that every entry in element_list is unique.
            if len(element_list) != len(set(element_list)):
                raise ValueError("Elements in element_list are not unique!")

            # Set the values for i_global.
            i = start_index
            i_nurbs_patch = 0
            for item in element_list:
                # As a NURBS patch can be defined with more elements, an offset is applied to the
                # rest of the items
                # TODO make i_global index-0 based
                item.i_global = i + 1
                if isinstance(item, _NURBSPatch):
                    item.n_nurbs_patch = i_nurbs_patch + 1
                    offset = item.get_number_elements()
                    i += offset
                    i_nurbs_patch += 1
                else:
                    i += 1

        def _dump_mesh_items(yaml_dict, section_name, data_list):
            """Output a section name and apply either the default dump or the
            specialized the dump_to_list for each list item."""

            # Do not write section if no content is available
            if len(data_list) == 0:
                return

            # Check if section already exists
            if section_name not in yaml_dict.keys():
                yaml_dict[section_name] = []

            item_dict_list = yaml_dict[section_name]
            for item in data_list:
                if hasattr(item, "dump_to_list"):
                    item_dict_list.extend(item.dump_to_list())
                elif isinstance(item, _BoundaryCondition):
                    item_dict_list.append(
                        {"E": item.geometry_set.i_global, **item.data}
                    )
                else:
                    raise TypeError(f"Could not dump {item}")

        # Add sets from couplings and boundary conditions to a temp container.
        mesh.unlink_nodes()
        start_indices_geometry_set = _get_global_start_geometry_set(self.sections)
        mesh_sets = mesh.get_unique_geometry_sets(
            geometry_set_start_indices=start_indices_geometry_set
        )

        # Assign global indices to all entries.
        start_index_nodes = _get_global_start_node(self.sections)
        _set_i_global(mesh.nodes, start_index=start_index_nodes)
        start_index_elements = _get_global_start_element(self.sections)
        _set_i_global_elements(mesh.elements, start_index=start_index_elements)
        start_index_materials = _get_global_start_material(self.sections)
        _set_i_global(mesh.materials, start_index=start_index_materials)
        start_index_functions = _get_global_start_function(self.sections)
        _set_i_global(mesh.functions, start_index=start_index_functions)

        # Add material data to the input file.
        _dump_mesh_items(self.sections, "MATERIALS", mesh.materials)

        # Add the functions.
        for function in mesh.functions:
            self.sections[f"FUNCT{function.i_global}"] = function.dump_to_list()

        # If there are couplings in the mesh, set the link between the nodes
        # and elements, so the couplings can decide which DOFs they couple,
        # depending on the type of the connected beam element.
        def get_number_of_coupling_conditions(key):
            """Return the number of coupling conditions in the mesh."""
            if (key, _mpy.geo.point) in mesh.boundary_conditions.keys():
                return len(mesh.boundary_conditions[key, _mpy.geo.point])
            else:
                return 0

        if (
            get_number_of_coupling_conditions(_mpy.bc.point_coupling)
            + get_number_of_coupling_conditions(_mpy.bc.point_coupling_penalty)
            > 0
        ):
            mesh.set_node_links()

        # Add the boundary conditions.
        for (bc_key, geom_key), bc_list in mesh.boundary_conditions.items():
            if len(bc_list) > 0:
                section_name = (
                    bc_key
                    if isinstance(bc_key, str)
                    else _boundary_condition_names[bc_key, geom_key]
                )
                _dump_mesh_items(self.sections, section_name, bc_list)

        # Add additional element sections, e.g., for NURBS knot vectors.
        for element in mesh.elements:
            element.dump_element_specific_section(self.sections)

        # Add the geometry sets.
        for geom_key, item in mesh_sets.items():
            if len(item) > 0:
                _dump_mesh_items(self.sections, _geometry_set_names[geom_key], item)

        # Add the nodes and elements.
        _dump_mesh_items(self.sections, "NODE COORDS", mesh.nodes)
        _dump_mesh_items(self.sections, "STRUCTURE ELEMENTS", mesh.elements)
        # TODO: reset all links and counters set in this method.

    def _get_header(self) -> dict:
        """Return the information header for the current MeshPy run.

        Returns:
            A dictionary with the header information.
        """

        header: dict = {"MeshPy": {}}

        header["MeshPy"]["creation_date"] = _datetime.now().isoformat(
            sep=" ", timespec="seconds"
        )

        # application which created the input file
        application_path = _Path(_sys.argv[0]).resolve()
        header["MeshPy"]["Application"] = {"path": str(application_path)}

        application_git_sha, application_git_date = _get_git_data(
            application_path.parent
        )
        if application_git_sha is not None and application_git_date is not None:
            header["MeshPy"]["Application"].update(
                {
                    "git_sha": application_git_sha,
                    "git_date": application_git_date,
                }
            )

        # MeshPy information
        meshpy_git_sha, meshpy_git_date = _get_git_data(
            _Path(__file__).resolve().parent
        )
        if meshpy_git_sha is not None and meshpy_git_date is not None:
            header["MeshPy"]["MeshPy"] = {
                "git_SHA": meshpy_git_sha,
                "git_date": meshpy_git_date,
            }

        # CubitPy information
        if _cubitpy_is_available():
            cubitpy_git_sha, cubitpy_git_date = _get_git_data(
                _os.path.dirname(_cubitpy.__file__)
            )

            if cubitpy_git_sha is not None and cubitpy_git_date is not None:
                header["MeshPy"]["CubitPy"] = {
                    "git_SHA": cubitpy_git_sha,
                    "git_date": cubitpy_git_date,
                }

        return header

    def _get_application_script(self, application_path: _Path) -> list[str]:
        """Get the script that created this input file.

        Args:
            application_path: Path to the script that created this input file.
        Returns:
            A list of strings with the script that created this input file.
        """

        application_script_lines = [
            "# Application script which created this input file:\n"
        ]

        with open(application_path) as script_file:
            application_script_lines.extend("# " + line for line in script_file)

        return application_script_lines
