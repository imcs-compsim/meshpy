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

import copy as _copy
import datetime as _datetime
import os as _os
import shutil as _shutil
import subprocess as _subprocess  # nosec B404
import sys as _sys
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional

import yaml as _yaml

import meshpy.core.conf as _conf
from meshpy.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from meshpy.core.conf import mpy as _mpy
from meshpy.core.container import ContainerBase as _ContainerBase
from meshpy.core.element import Element as _Element
from meshpy.core.geometry_set import GeometryName as _GeometryName
from meshpy.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.node import Node as _Node
from meshpy.core.nurbs_patch import NURBSPatch as _NURBSPatch
from meshpy.utils.environment import cubitpy_is_available as _cubitpy_is_available
from meshpy.utils.environment import fourcipp_is_available as _fourcipp_is_available

if _cubitpy_is_available():
    import cubitpy as _cubitpy


def _get_geometry_set_indices_from_section(
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
        index_geometry_set = int(line.split()[-1])
        index_node = int(line.split()[1]) - 1
        if index_geometry_set not in geometry_set_dict:
            geometry_set_dict[index_geometry_set] = []
        if append_node_ids:
            geometry_set_dict[index_geometry_set].append(index_node)

    return geometry_set_dict


def _get_yaml_geometry_sets(
    nodes: _List[_Node], geometry_key: _conf.Geometry, section_list: _List
):
    """Add sets of points, lines, surfaces or volumes to the object."""

    # Create the individual geometry sets. The nodes are still integers at this
    # point. They have to be converted to links to the actual nodes later on.
    geometry_set_dict = _get_geometry_set_indices_from_section(section_list)
    geometry_sets_in_this_section = []
    for node_ids in geometry_set_dict.values():
        geometry_sets_in_this_section.append(
            _GeometrySetNodes(
                geometry_key, nodes=[nodes[node_id] for node_id in node_ids]
            )
        )
    return geometry_sets_in_this_section


class InputFile(_Mesh):
    """An item that represents a complete 4C input file."""

    # Define the names of sections and boundary conditions in the input file.
    geometry_set_names = {
        _mpy.geo.point: "DNODE-NODE TOPOLOGY",
        _mpy.geo.line: "DLINE-NODE TOPOLOGY",
        _mpy.geo.surface: "DSURF-NODE TOPOLOGY",
        _mpy.geo.volume: "DVOL-NODE TOPOLOGY",
    }
    boundary_condition_names = {
        (_mpy.bc.dirichlet, _mpy.geo.point): "DESIGN POINT DIRICH CONDITIONS",
        (_mpy.bc.dirichlet, _mpy.geo.line): "DESIGN LINE DIRICH CONDITIONS",
        (_mpy.bc.dirichlet, _mpy.geo.surface): "DESIGN SURF DIRICH CONDITIONS",
        (_mpy.bc.dirichlet, _mpy.geo.volume): "DESIGN VOL DIRICH CONDITIONS",
        (_mpy.bc.locsys, _mpy.geo.point): "DESIGN POINT LOCSYS CONDITIONS",
        (_mpy.bc.locsys, _mpy.geo.line): "DESIGN LINE LOCSYS CONDITIONS",
        (_mpy.bc.locsys, _mpy.geo.surface): "DESIGN SURF LOCSYS CONDITIONS",
        (_mpy.bc.locsys, _mpy.geo.volume): "DESIGN VOL LOCSYS CONDITIONS",
        (_mpy.bc.neumann, _mpy.geo.point): "DESIGN POINT NEUMANN CONDITIONS",
        (_mpy.bc.neumann, _mpy.geo.line): "DESIGN LINE NEUMANN CONDITIONS",
        (_mpy.bc.neumann, _mpy.geo.surface): "DESIGN SURF NEUMANN CONDITIONS",
        (_mpy.bc.neumann, _mpy.geo.volume): "DESIGN VOL NEUMANN CONDITIONS",
        (
            _mpy.bc.moment_euler_bernoulli,
            _mpy.geo.point,
        ): "DESIGN POINT MOMENT EB CONDITIONS",
        (
            _mpy.bc.beam_to_solid_volume_meshtying,
            _mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE",
        (
            _mpy.bc.beam_to_solid_volume_meshtying,
            _mpy.geo.volume,
        ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME",
        (
            _mpy.bc.beam_to_solid_surface_meshtying,
            _mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING LINE",
        (
            _mpy.bc.beam_to_solid_surface_meshtying,
            _mpy.geo.surface,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE",
        (
            _mpy.bc.beam_to_solid_surface_contact,
            _mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT LINE",
        (
            _mpy.bc.beam_to_solid_surface_contact,
            _mpy.geo.surface,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE",
        (_mpy.bc.point_coupling, _mpy.geo.point): "DESIGN POINT COUPLING CONDITIONS",
        (
            _mpy.bc.beam_to_beam_contact,
            _mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS",
        (
            _mpy.bc.point_coupling_penalty,
            _mpy.geo.point,
        ): "DESIGN POINT PENALTY COUPLING CONDITIONS",
        (
            "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
            _mpy.geo.surface,
        ): "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
    }

    def __init__(self, *, yaml_file: _Optional[_Path] = None, cubit=None):
        """Initialize the input file.

        Args:
            yaml_file:
                A file path to an existing input file that will be read into this
                object.
            cubit:
                A cubit object, that contains an input file which will be loaded
                into this input file.
        """

        super().__init__()

        # Everything that is not a full MeshPy object is stored here, e.g., parameters
        # and imported nodes/elements/materials/...
        self.sections: _Dict[str, _Any] = dict()

        # Contents of NOX xml file.
        self.nox_xml = None
        self._nox_xml_file = None

        if yaml_file is not None:
            self.read_yaml(yaml_file)

        # if cubit is not None:
        #     self._read_dat_lines(cubit.get_dat_lines())

    def read_yaml(self, file_path: _Path):
        """Read an existing input file into this object.

        Args:
            file_path:
                A file path to an existing input file that will be read into this
                object.
        """

        if _fourcipp_is_available():
            raise ValueError("Use fourcipp to parse the yaml file.")

        with open(file_path) as stream:
            self.sections = _yaml.safe_load(stream)

        if _mpy.import_mesh_full:
            self.sections_to_mesh()

    def sections_to_mesh(self):
        """Convert mesh items, e.g., nodes, elements, element sets, node sets,
        boundary conditions, materials, ...

        to "true" MeshPy objects.
        """

        def _get_section_items(section_name):
            """Return the items in a given section.

            Since we will add the created MeshPy objects to the mesh, we
            delete them from the plain data storage to avoid having
            duplicate entries.
            """
            if section_name in self.sections:
                return_list = self.sections[section_name]
                self.sections[section_name] = []
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
                raise ValueError(
                    "Port this functionality to not use the legacy string."
                )

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

            mesh.elements.append(_Element.from_legacy_string(element_nodes, item))

        # Add geometry sets
        for section_name in self.sections.keys():
            if section_name.endswith("TOPOLOGY"):
                section_items = _get_section_items(section_name)
                if len(section_items) > 0:
                    # Get the geometry key for this set
                    for key, value in self.geometry_set_names.items():
                        if value == section_name:
                            geometry_key = key
                            break
                    else:
                        raise ValueError(f"Could not find the set {section_name}")
                    geometry_sets_in_this_section = _get_yaml_geometry_sets(
                        mesh.nodes, geometry_key, section_items
                    )
                    mesh.geometry_sets[geometry_key] = geometry_sets_in_this_section

        # Add boundary conditions
        for (
            bc_key,
            geometry_key,
        ), section_name in self.boundary_condition_names.items():
            for item in _get_section_items(section_name):
                mesh.boundary_conditions.append(
                    (bc_key, geometry_key),
                    _BoundaryConditionBase.from_dict(
                        mesh.geometry_sets[geometry_key], bc_key, item
                    ),
                )

        self.add(mesh)

    def add(self, *args, **kwargs):
        """Add to this object.

        If the type is not recognized, the child add method is called.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            # TODO: We have to check here if the item is not of any of the types we
            # use that derive from dict, as they should be added in super().add
            if not isinstance(args[0], _ContainerBase) and not isinstance(
                args[0], _GeometryName
            ):
                self.add_section(args[0], **kwargs)
                return

        super().add(*args, **kwargs)

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
        self, file_path: _Path, *, nox_xml_file: _Optional[str] = None, **kwargs
    ):
        """Write the input file to disk.

        Args:
            file_path: str
                Path to the input file that should be created.
            nox_xml_file: str
                (optional) If this argument is given, the xml file will be created
                with this name, in the same directory as the input file.
        """

        # Check if a xml file needs to be written.
        if self.nox_xml is not None:
            if nox_xml_file is None:
                # Get the name of the xml file.
                self._nox_xml_file = (
                    _os.path.splitext(_os.path.basename(file_path))[0] + ".xml"
                )
            else:
                self._nox_xml_file = nox_xml_file

            # Write the xml file to the disc.
            with open(
                _os.path.join(_os.path.dirname(file_path), self._nox_xml_file), "w"
            ) as xml_file:
                xml_file.write(self.nox_xml)

        with open(file_path, "w") as input_file:
            data = self.get_dict_to_dump(**kwargs)
            _yaml.dump(data, input_file, width=float("inf"))

    def get_dict_to_dump(
        self,
        *,
        header: bool = True,
        dat_header: bool = True,
        add_script_to_header: bool = True,
        check_nox: bool = True,
    ):
        """Return the dictionary representation of this input file for dumping
        to a yaml file.

        Args:
            header:
                If the header should be exported to the input file files.
            dat_header:
                If header lines from the imported dat file should be exported.
            append_script_to_header:
                If true, a copy of the executing script will be added to the input
                file. This is only in affect when dat_header is True.
            check_nox:
                If this is true, an error will be thrown if no nox file is set.
        """

        # Perform some checks on the mesh.
        if _mpy.check_overlapping_elements:
            self.check_overlapping_elements()

        # The base dictionary we use here is the one that already exists.
        # This one might already contain mesh sections - stored in pure
        # data format.
        # TODO: Check if the deepcopy makes sense to be optional
        yaml_dict = _copy.deepcopy(self.sections)

        # TODO: Add header to yaml
        # # Add header to the input file.
        # end_text = None

        # lines.extend(["// " + line for line in _mpy.input_file_meshpy_header])
        # if header:
        #     header_text, end_text = self._get_header(add_script_to_header)
        #     lines.append(header_text)
        # if dat_header:
        #     lines.extend(self.dat_header)

        # Check if a file has to be created for the NOX xml information.
        if self.nox_xml is not None:
            if self._nox_xml_file is None:
                if check_nox:
                    raise ValueError("NOX xml content is given, but no file defined!")
                nox_xml_name = "NOT_DEFINED"
            else:
                nox_xml_name = self._nox_xml_file
            # TODO: Use something like the add_section here
            yaml_dict["STRUCT NOX/Status Test"] = {"XML File": nox_xml_name}

        def _get_global_start_geometry_set(yaml_dict):
            """Get the indices for the first "real" MeshPy geometry sets."""

            start_indices_geometry_set = {}
            for geometry_type, section_name in self.geometry_set_names.items():
                max_geometry_set_id = 0
                if section_name in yaml_dict:
                    section_list = yaml_dict[section_name]
                    if len(section_list) > 0:
                        geometry_set_dict = _get_geometry_set_indices_from_section(
                            section_list, append_node_ids=False
                        )
                        max_geometry_set_id = max(geometry_set_dict.keys())
                start_indices_geometry_set[geometry_type] = max_geometry_set_id
            return start_indices_geometry_set

        def _get_global_start_node(yaml_dict):
            """Get the index for the first "real" MeshPy node."""

            if _fourcipp_is_available():
                raise ValueError(
                    "Port this functionality to not use the legacy format any more"
                    "TODO: Check if we really want this - should we just assume that the"
                    "imported nodes are in order and without any 'missing' nodes?"
                )

            section_name = "NODE COORDS"
            if section_name in yaml_dict:
                return len(yaml_dict[section_name])
            else:
                return 0

        def _get_global_start_element(yaml_dict):
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
                if section_name in yaml_dict:
                    start_index += len(yaml_dict[section_name])
            return start_index

        def _get_global_start_material(yaml_dict):
            """Get the index for the first "real" MeshPy material.

            We have to account for materials imported from yaml files
            that have arbitrary numbering.
            """

            # Get the maximum material index in materials imported from a yaml file
            max_material_id = 0
            section_name = "MATERIALS"
            if section_name in yaml_dict:
                for material in yaml_dict[section_name]:
                    max_material_id = max(max_material_id, material["MAT"])
            return max_material_id

        def _get_global_start_function(yaml_dict):
            """Get the index for the first "real" MeshPy function."""

            max_function_id = 0
            for section_name in yaml_dict.keys():
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
            """Output a section name and apply the dump_to_list for each list
            item."""

            # Do not write section if no content is available
            if len(data_list) == 0:
                return

            # Check if section already exists
            if section_name not in yaml_dict.keys():
                yaml_dict[section_name] = []

            item_dict_list = yaml_dict[section_name]
            for item in data_list:
                item_dict_list.extend(item.dump_to_list())

        # Add sets from couplings and boundary conditions to a temp container.
        self.unlink_nodes()
        start_indices_geometry_set = _get_global_start_geometry_set(yaml_dict)
        mesh_sets = self.get_unique_geometry_sets(
            geometry_set_start_indices=start_indices_geometry_set
        )

        # Assign global indices to all entries.
        start_index_nodes = _get_global_start_node(yaml_dict)
        _set_i_global(self.nodes, start_index=start_index_nodes)
        start_index_elements = _get_global_start_element(yaml_dict)
        _set_i_global_elements(self.elements, start_index=start_index_elements)
        start_index_materials = _get_global_start_material(yaml_dict)
        _set_i_global(self.materials, start_index=start_index_materials)
        start_index_functions = _get_global_start_function(yaml_dict)
        _set_i_global(self.functions, start_index=start_index_functions)

        # Add material data to the input file.
        _dump_mesh_items(yaml_dict, "MATERIALS", self.materials)

        # Add the functions.
        for function in self.functions:
            yaml_dict[f"FUNCT{function.i_global}"] = function.dump_to_list()

        # If there are couplings in the mesh, set the link between the nodes
        # and elements, so the couplings can decide which DOFs they couple,
        # depending on the type of the connected beam element.
        def get_number_of_coupling_conditions(key):
            """Return the number of coupling conditions in the mesh."""
            if (key, _mpy.geo.point) in self.boundary_conditions.keys():
                return len(self.boundary_conditions[key, _mpy.geo.point])
            else:
                return 0

        if (
            get_number_of_coupling_conditions(_mpy.bc.point_coupling)
            + get_number_of_coupling_conditions(_mpy.bc.point_coupling_penalty)
            > 0
        ):
            self.set_node_links()

        # Add the boundary conditions.
        for (bc_key, geom_key), bc_list in self.boundary_conditions.items():
            if len(bc_list) > 0:
                section_name = (
                    bc_key
                    if isinstance(bc_key, str)
                    else self.boundary_condition_names[bc_key, geom_key]
                )
                _dump_mesh_items(yaml_dict, section_name, bc_list)

        # Add additional element sections (e.g. STRUCTURE KNOTVECTORS)
        # We only need to to this on the "real" elements as the imported ones already have their
        # dat sections.
        for element in self.elements:
            element.dump_element_specific_section(yaml_dict)

        # Add the geometry sets.
        for geom_key, item in mesh_sets.items():
            if len(item) > 0:
                _dump_mesh_items(yaml_dict, self.geometry_set_names[geom_key], item)

        # Add the nodes and elements.
        _dump_mesh_items(yaml_dict, "NODE COORDS", self.nodes)
        _dump_mesh_items(yaml_dict, "STRUCTURE ELEMENTS", self.elements)

        # TODO: what to do here - how to add the script
        # # Add end text.
        # if end_text is not None:
        #     lines.append(end_text)

        return yaml_dict

    def _get_header(self, add_script):
        """Return the header for the input file."""

        def get_git_data(repo):
            """Return the hash and date of the current git commit."""
            git = _shutil.which("git")
            if git is None:
                raise RuntimeError("Git executable not found")
            out_sha = _subprocess.run(  # nosec B603
                [git, "rev-parse", "HEAD"],
                cwd=repo,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.DEVNULL,
            )
            out_date = _subprocess.run(  # nosec B603
                [git, "show", "-s", "--format=%ci"],
                cwd=repo,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.DEVNULL,
            )
            if not out_sha.returncode + out_date.returncode == 0:
                return None, None
            else:
                sha = out_sha.stdout.decode("ascii").strip()
                date = out_date.stdout.decode("ascii").strip()
                return sha, date

        headers = []
        end_text = None

        # Header containing model information.
        current_time_string = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_header = f"// Date:       {current_time_string}\n"
        if self.description:
            model_header += f"// Description: {self.description}\n"
        headers.append(model_header)

        # Get information about the script.
        script_path = _os.path.realpath(_sys.argv[0])
        script_git_sha, script_git_date = get_git_data(_os.path.dirname(script_path))
        script_header = "// Script used to create input file:\n"
        script_header += f"// path:       {script_path}\n"
        if script_git_sha is not None:
            script_header += (
                f"// git sha:    {script_git_sha}\n// git date:   {script_git_date}\n"
            )
        headers.append(script_header)

        # Header containing meshpy information.
        meshpy_git_sha, meshpy_git_date = get_git_data(
            _os.path.dirname(_os.path.realpath(__file__))
        )
        headers.append(
            "// Input file created with meshpy\n"
            f"// git sha:    {meshpy_git_sha}\n"
            f"// git date:   {meshpy_git_date}\n"
        )

        if _cubitpy_is_available():
            # Get git information about cubitpy.
            cubitpy_git_sha, cubitpy_git_date = get_git_data(
                _os.path.dirname(_cubitpy.__file__)
            )

            if cubitpy_git_sha is not None:
                # Cubitpy_header.
                headers.append(
                    "// The module cubitpy was loaded\n"
                    f"// git sha:    {cubitpy_git_sha}\n"
                    f"// git date:   {cubitpy_git_date}\n"
                )

        string_line = "// " + "".join(["-" for _i in range(_mpy.dat_len_section - 3)])

        # If needed, append the contents of the script.
        if add_script:
            # Header for the script 'section'.
            script_lines = [
                string_line
                + "\n// Full script used to create this input file.\n"
                + string_line
                + "\n"
            ]

            # Get the contents of script.
            with open(script_path) as script_file:
                script_lines.extend(script_file.readlines())

            # Comment the python code lines.
            end_text = "//".join(script_lines)

        return (
            string_line + "\n" + (string_line + "\n").join(headers) + string_line
        ), end_text
