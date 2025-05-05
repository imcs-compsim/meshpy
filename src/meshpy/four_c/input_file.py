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
from typing import Tuple as _Tuple
from typing import Union as _Union

from fourcipp.fourc_input import FourCInput as _FourCInput

from meshpy.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from meshpy.core.conf import mpy as _mpy
from meshpy.core.coupling import Coupling as _Coupling
from meshpy.core.element import Element as _Element
from meshpy.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from meshpy.core.material import Material as _Material
from meshpy.core.mesh import Mesh as _Mesh
from meshpy.core.node import Node as _Node
from meshpy.four_c.function import Function as _Function
from meshpy.four_c.input_file_mappings import (
    FOUR_C_INPUT_FILE_MAPPINGS as _FOUR_C_INPUT_FILE_MAPPINGS,
)
from meshpy.utils.convert_mesh_to_dict import (
    convert_mesh_to_dict as _convert_mesh_to_dict,
)
from meshpy.utils.environment import cubitpy_is_available as _cubitpy_is_available
from meshpy.utils.environment import get_git_data as _get_git_data

if _cubitpy_is_available():
    import cubitpy as _cubitpy


class FourCInputFile(_FourCInput):
    """FourC input file class."""

    @classmethod
    def from_4C_yaml(
        cls,
        input_file_path: _Path,
        convert_input_to_mesh: bool = True,
    ) -> _Union[_FourCInput, _Tuple[_FourCInput, _Mesh]]:
        """Load a 4C yaml file and optionally convert the contents into a
        MeshPy mesh.

        Args:
            input_file_path: Path to yaml file
            convert_input_to_mesh: Boolean option to either convert the input to a
                MeshPy mesh or not. Defaults to True.

        Returns:
            A fourcipp input file object and optionally a MeshPy mesh object if
                convert_input_to_mesh is True.
        """

        input_file = super().from_4C_yaml(input_file_path, header_only=False)

        if not convert_input_to_mesh:
            return input_file

        mesh = _Mesh()

        # convert nodes to mesh
        if "NODE COORDS" in input_file:
            mesh.nodes = [
                _Node(node["COORD"]) for node in input_file.pop("NODE COORDS")
            ]

        # convert elements to mesh
        if "STRUCTURE ELEMENTS" in input_file:
            for element in input_file.pop("STRUCTURE ELEMENTS"):
                nodes = [
                    mesh.nodes[node_id - 1]
                    for node_id in element["cell"]["connectivity"]
                ]
                mesh.elements.append(
                    _Element.from_fourcipp_dict(nodes=nodes, element=element)
                )

        # convert geometry sets to mesh
        extracted_geometry_sets = {
            section: input_file.sections.pop(section)
            for section in list(input_file.sections)
            if section
            in list(_FOUR_C_INPUT_FILE_MAPPINGS["geometry_set_names"].values())
        }

        for section_name, geometry_set in extracted_geometry_sets.items():
            geometry_type = next(
                type
                for type, geometry_set_name in _FOUR_C_INPUT_FILE_MAPPINGS[
                    "geometry_set_names"
                ].items()
                if geometry_set_name == section_name
            )

            mesh.geometry_sets[geometry_type] = []
            for node_set in geometry_set:
                mesh.geometry_sets[geometry_type].append(
                    _GeometrySetNodes(
                        geometry_type=geometry_type,
                        nodes=[mesh.nodes[node_id] for node_id in node_set["node_id"]],
                    )
                )

        # convert boundary conditions to mesh
        extracted_boundary_conditions = {
            section: input_file.sections.pop(section)
            for section in list(input_file.sections)
            if section
            in list(_FOUR_C_INPUT_FILE_MAPPINGS["boundary_condition_names"].values())
        }

        for section_name, bc_data in extracted_boundary_conditions.items():
            (bc_key, geometry_key) = next(
                item
                for item, section in _FOUR_C_INPUT_FILE_MAPPINGS[
                    "boundary_condition_names"
                ].items()
                if section == section_name
            )

            bc_geometry_set = next(
                geometry_set
                for geometry_set in mesh.geometry_sets[geometry_key]
                if geometry_set.dump_to_list()[0]["d_id"] == bc_data["E"]
            )

            if bc_key == _mpy.bc.point_coupling:
                mesh.boundary_conditions.append(
                    (bc_key, geometry_key),
                    _Coupling(
                        bc_geometry_set,
                        bc_key,
                        bc_data.pop("E", None),
                        check_overlapping_nodes=False,
                    ),
                )
            elif bc_key in [
                _mpy.bc.dirichlet,
                _mpy.bc.neumann,
                _mpy.bc.locsys,
                _mpy.bc.beam_to_solid_surface_meshtying,
                _mpy.bc.beam_to_solid_surface_contact,
                _mpy.bc.beam_to_solid_volume_meshtying,
            ]:
                mesh.boundary_conditions.append(
                    (bc_key, geometry_key),
                    _BoundaryCondition(
                        bc_geometry_set, bc_key, bc_data.pop("E", None), bc_type=bc_key
                    ),
                )
            else:
                raise ValueError(
                    f"Unable to convert the following boundary condition `{section_name}` into a MeshPy mesh object."
                )

        return input_file, mesh

    def add(self, item: _Union[_Mesh, dict, _FourCInput]) -> None:
        """Add contents to the input file.

        If a MeshPy mesh is provided, it will be converted to a dictionary and added to the
        input file. All other file types (e.g., dictionaries, fourcipp objects) will be added via the super method.

        Args:
            item: The item to be added to the input file.
        """

        if isinstance(item, _Mesh):
            # convert mesh to dictionary
            item = _convert_mesh_to_dict(
                item,
                global_start_indices=self._get_global_start_indices(
                    [_Node, _Element, _Material, _Function]
                    + list(_FOUR_C_INPUT_FILE_MAPPINGS["geometry_set_names"].keys()),
                ),
                input_file_mappings=_FOUR_C_INPUT_FILE_MAPPINGS,
            )

        super().add(item)

    def _get_global_start_indices(self, item_types: list) -> dict:
        """Get the global start index for a list of given item types.

        Args:
            item_type: List of item types (e.g., Node, Element, Material, Function, ...).

        Returns:
            Dictionary with the global start index for each item type.
        """

        indices: dict = {}

        for item_type in item_types:
            # nodes
            if item_type == _Node:
                indices[_Node] = len(self.sections.get("NODE COORDS", []))

            # elements
            elif item_type == _Element:
                indices[_Element] = sum(
                    len(self.sections.get(section, []))
                    for section in ["FLUID ELEMENTS", "STRUCTURE ELEMENTS"]
                )

            # materials
            elif item_type == _Material:
                indices[_Material] = max(
                    (
                        material["MAT"]
                        for material in self.sections.get("MATERIALS", [])
                    ),
                    default=0,
                )

            # functions
            elif item_type == _Function:
                indices[_Function] = max(
                    (
                        int(name.split("FUNCT")[-1])
                        for name in self.sections
                        if name.startswith("FUNCT")
                    ),
                    default=0,
                )

            # d-topologies (DNODE, DLINE, DSURF, DVOL)
            elif item_type in _FOUR_C_INPUT_FILE_MAPPINGS["geometry_set_names"].keys():
                indices[item_type] = max(
                    (
                        d_topology["d_id"]
                        for d_topology in self.sections.get(
                            _FOUR_C_INPUT_FILE_MAPPINGS["geometry_set_names"][
                                item_type
                            ],
                            [],
                        )
                    ),
                    default=0,
                )

            else:
                raise ValueError(
                    f"Global start index determination for {item_type} not available!"
                )

        return indices

    def dump(
        self,
        file_path: _Path,
        *,
        sort_sections: bool = False,
        validate_input_file: bool = True,
        add_header_information: bool = True,
        add_header_default: bool = True,
        add_footer_application_script: bool = True,
    ):
        """Dump object to yaml file.

        Args:
            file_path: Path to the file.
            sort_sections: Boolean option to alphabeticall sort the sections in
                the yaml file. Defaults to False.
            validate_input_file: Boolean option to validate the input file based
                on the 4C metadata file. Defaults to True.
            add_header_information: Boolean option to add header information
                to the yaml file. Defaults to True.
            add_header_default: Boolean option to add default MeshPy header
                information (authors, copyright, authors, etc.) to the yaml file.
                Defaults to True.
            add_footer_application_script: Boolean option to add the application
                script that created this input file to the end of the file.
                Defaults to True.
        """

        if add_header_information:
            self.add(self._get_header())

        # TODO fix this check
        # if _mpy.check_overlapping_elements:
        #     check_overlapping_elements() # function in utilities
        #     pass

        super().dump(
            file_path, sort_sections=sort_sections, validate=validate_input_file
        )

        if add_header_default or add_footer_application_script:
            with open(file_path, "r") as input_file:
                lines = input_file.readlines()

            if add_header_default:
                lines = [
                    "# " + line + "\n" for line in _mpy.input_file_meshpy_header
                ] + lines

            if add_footer_application_script:
                application_path = _Path(_sys.argv[0]).resolve()
                lines += self._get_application_script(application_path)

            with open(file_path, "w") as input_file:
                input_file.writelines(lines)

    def _get_header(self) -> dict:
        """Return the information header for the current MeshPy run.

        Returns:
            A dictionary with the header information.
        """

        metadata_section_name = _FOUR_C_INPUT_FILE_MAPPINGS["metadata_section_name"]

        header: dict = {metadata_section_name: {"MeshPy": {}}}

        header[metadata_section_name]["MeshPy"]["creation_date"] = (
            _datetime.now().isoformat(sep=" ", timespec="seconds")
        )

        # application which created the input file
        application_path = _Path(_sys.argv[0]).resolve()
        header[metadata_section_name]["MeshPy"]["Application"] = {
            "path": str(application_path)
        }

        application_git_sha, application_git_date = _get_git_data(
            application_path.parent
        )
        if application_git_sha is not None and application_git_date is not None:
            header[metadata_section_name]["MeshPy"]["Application"].update(
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
            header[metadata_section_name]["MeshPy"]["MeshPy"] = {
                "git_SHA": meshpy_git_sha,
                "git_date": meshpy_git_date,
            }

        # CubitPy information
        if _cubitpy_is_available():
            cubitpy_git_sha, cubitpy_git_date = _get_git_data(
                _os.path.dirname(_cubitpy.__file__)
            )

            if cubitpy_git_sha is not None and cubitpy_git_date is not None:
                header[metadata_section_name]["MeshPy"]["CubitPy"] = {
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
