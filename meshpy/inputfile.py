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
"""This module defines the classes that are used to create an input file for
4C."""

import datetime
import os
import re
import sys

from .base_mesh_item import BaseMeshItemFull, BaseMeshItemString
from .boundary_condition import BoundaryConditionBase
from .conf import mpy
from .container import BoundaryConditionContainer, GeometrySetContainer
from .element import Element
from .geometry_set import GeometrySetNodes
from .mesh import Mesh
from .node import Node
from .nurbs_patch import NURBSPatch
from .utility import get_git_data


def get_section_string(section_name):
    """Return the string for a section in the dat file."""
    return (
        "".join(["-" for _i in range(mpy.dat_len_section - len(section_name))])
        + section_name
    )


class InputLine:
    """This class is a single option in a 4C input file."""

    def __init__(self, *args, option_comment=None, option_overwrite=False):
        """Set a line of the 4C input file.

        Args
        ----
        args: str
            First the string is checked for a comment at the end of it. Then
            the remaining string will be searched for an equal sign and if
            found split up at that sign. Otherwise it will be checked how many
            parts separated by spaces there are in the string. If there are
            exactly two parts, the first one will be the option_name the second
            one the option_value.
        args: (str, str)
            The first one will be the option_name the second one the
            option_value.
        option_comment: str
            This will be added as a comment to this line.
        option_overwrite: bool
            If this object is added to a section which already contains an
            option with the same name, this flag decides weather the option
            will be overwritten.
        """

        self.option_name = ""
        self.option_value = ""
        self.option_comment = ""
        self.option_value_pad = "  "
        self.overwrite = option_overwrite

        if len(args) == 1:
            # Set from single a string.
            string = args[0]

            # First check if the line has a comment.
            first_comment = string.find("//")
            if not first_comment == -1:
                self.option_comment = string[first_comment:]
                string = string[:first_comment]

            # Check if there is an equal sign in the string.
            if not string.find("=") == -1:
                string_split = [text.strip() for text in string.split("=")]
                self.option_value_pad = "= "
            elif len(string.split()) == 2:
                string_split = [text.strip() for text in string.split()]
            else:
                string_split = ["", ""]
                self.option_comment = args[0].strip()
        else:
            string_split = [str(arg).strip() for arg in args]

        if option_comment is not None:
            if self.option_comment == "":
                self.option_comment = f"// {option_comment}"
            else:
                self.option_comment += f" // {option_comment}"

        # Check if the string_split array has a suitable size.
        if len(string_split) == 2:
            self.option_name = string_split[0]
            self.option_value = string_split[1]
        else:
            raise ValueError(
                "Could not process the input parameters:"
                f"\nargs:\n    {args}\noption_comment:\n    {option_comment}"
            )

    def get_key(self):
        """Return a key that will be used in the dictionary storage for this
        item.

        If the option_comment is empty the identifier of this object
        will be returned, so than more than one empty lines can be in
        one section.
        """

        if self.option_name == "":
            if self.option_comment == "":
                return str(id(self))
            else:
                return self.option_comment
        else:
            return self.option_name

    def __str__(self):
        """Return the string for this line of the input file."""
        string = ""
        if self.option_name != "":
            string += "{:<35} {}{}".format(
                self.option_name, self.option_value_pad, self.option_value
            )
        if self.option_comment != "":
            if self.option_name != "":
                string += " "
            string += f"{self.option_comment}"
        return string


class InputSection:
    """Represent a single section in the input file."""

    def __init__(self, name, *args, **kwargs):
        # Section title.
        self.name = name

        # Each input line will be one entry in this dictionary.
        self.data = dict()

        for arg in args:
            self.add(arg, **kwargs)

    def add(self, data, **kwargs):
        """Add data to this section in the form of an InputLine object.

        Args
        ----
        data: str, list(str)
            If data is a string, it will be split up into lines.
            Each line will be added as an InputLine object.
        """

        if isinstance(data, str):
            data_lines = data.split("\n")
        else:
            # Check if data has entries
            if len(data) == 0:
                return
            data_lines = data

        # Remove the first and or last line if it is empty.
        for index in [0, -1]:
            if data_lines[index].strip() == "":
                del data_lines[index]

        # Add the data lines.
        for item in data_lines:
            self._add_data(InputLine(item, **kwargs))

    def _add_data(self, option):
        """Add a InputLine object to the item."""

        if (option.get_key() not in self.data.keys()) or option.overwrite:
            self.data[option.get_key()] = option
        else:
            raise KeyError(f"Key {option.get_key()} is already set!")

    def merge_section(self, section):
        """Merge this section with another.

        This one is the master.
        """

        for option in section.data.values():
            self._add_data(option)

    def get_dat_lines(self):
        """Return the lines for this section in the input file."""

        lines = [get_section_string(self.name)]
        lines.extend([str(line) for line in self.data.values()])
        return lines


class InputSectionMultiKey(InputSection):
    """Represent a single section in the input file.

    This section can have the same key multiple times.
    """

    def _add_data(self, option):
        """Add an InputLine object to the item.

        Each key can exist multiple times.
        """

        # We add each line with a key that represents the index of the line.
        # This would be better with a list, but by doing it his way we can use
        # the same structure as in the base class.
        self.data[len(self.data)] = option


class InputFile(Mesh):
    """An item that represents a complete 4C input file."""

    # Define the names of sections and boundary conditions in the input file.
    geometry_set_names = {
        mpy.geo.point: "DNODE-NODE TOPOLOGY",
        mpy.geo.line: "DLINE-NODE TOPOLOGY",
        mpy.geo.surface: "DSURF-NODE TOPOLOGY",
        mpy.geo.volume: "DVOL-NODE TOPOLOGY",
    }
    boundary_condition_names = {
        (mpy.bc.dirichlet, mpy.geo.point): "DESIGN POINT DIRICH CONDITIONS",
        (mpy.bc.dirichlet, mpy.geo.line): "DESIGN LINE DIRICH CONDITIONS",
        (mpy.bc.dirichlet, mpy.geo.surface): "DESIGN SURF DIRICH CONDITIONS",
        (mpy.bc.dirichlet, mpy.geo.volume): "DESIGN VOL DIRICH CONDITIONS",
        (mpy.bc.locsys, mpy.geo.point): "DESIGN POINT LOCSYS CONDITIONS",
        (mpy.bc.locsys, mpy.geo.line): "DESIGN LINE LOCSYS CONDITIONS",
        (mpy.bc.locsys, mpy.geo.surface): "DESIGN SURF LOCSYS CONDITIONS",
        (mpy.bc.locsys, mpy.geo.volume): "DESIGN VOL LOCSYS CONDITIONS",
        (mpy.bc.neumann, mpy.geo.point): "DESIGN POINT NEUMANN CONDITIONS",
        (mpy.bc.neumann, mpy.geo.line): "DESIGN LINE NEUMANN CONDITIONS",
        (mpy.bc.neumann, mpy.geo.surface): "DESIGN SURF NEUMANN CONDITIONS",
        (mpy.bc.neumann, mpy.geo.volume): "DESIGN VOL NEUMANN CONDITIONS",
        (
            mpy.bc.moment_euler_bernoulli,
            mpy.geo.point,
        ): "DESIGN POINT MOMENT EB CONDITIONS",
        (
            mpy.bc.beam_to_solid_volume_meshtying,
            mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE",
        (
            mpy.bc.beam_to_solid_volume_meshtying,
            mpy.geo.volume,
        ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME",
        (
            mpy.bc.beam_to_solid_surface_meshtying,
            mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING LINE",
        (
            mpy.bc.beam_to_solid_surface_meshtying,
            mpy.geo.surface,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE",
        (
            mpy.bc.beam_to_solid_surface_contact,
            mpy.geo.line,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT LINE",
        (
            mpy.bc.beam_to_solid_surface_contact,
            mpy.geo.surface,
        ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE",
        (mpy.bc.point_coupling, mpy.geo.point): "DESIGN POINT COUPLING CONDITIONS",
        (
            mpy.bc.point_coupling_penalty,
            mpy.geo.point,
        ): "DESIGN POINT PENALTY COUPLING CONDITIONS",
        (
            "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
            mpy.geo.surface,
        ): "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
    }

    # Sections that won't be exported to input file.
    skip_sections = [
        "ALE ELEMENTS",
        "LUBRICATION ELEMENTS",
        "TRANSPORT2 ELEMENTS",
        "ACOUSTIC ELEMENTS",
        "CELL ELEMENTS",
        "CELLSCATRA ELEMENTS",
        "ARTERY ELEMENTS",
        "ELECTROMAGNETIC ELEMENTS",
    ]

    def __init__(self, *, description=None, dat_file=None, cubit=None):
        """Initialize the input file.

        Args
        ----
        description: srt
            Will be shown in the input file as description of the system.
        dat_file: str
            A file path to an existing input file that will be read into this
            object.
        cubit:
            A cubit object, that contains an input file. The input lines are
            loaded with the get_dat_lines method. Mutually exclusive with the
            option dat_file.
        """

        super().__init__()

        self.description = description
        self.dat_header = []

        # In case we import a 4C dat file with plain string data, we store them in these lists,
        # they do not interfere with other operations.
        self.dat_nodes = []
        self.dat_elements = []
        self.dat_elements_fluid = []
        self.dat_geometry_sets = GeometrySetContainer()
        self.dat_boundary_conditions = BoundaryConditionContainer()

        # Contents of NOX xml file.
        self.nox_xml = None
        self._nox_xml_file = None

        # Dictionaries for sections other than mesh sections.
        self.sections = dict()

        # Flag if dat file was loaded.
        self._dat_file_loaded = False

        # Load existing dat files. If both of the following if statements are
        # true, an error will be thrown.
        if dat_file is not None:
            self.read_dat(dat_file)
        if cubit is not None:
            self._read_dat_lines(cubit.get_dat_lines())

    def read_dat(self, file_path):
        """Read an existing input file into this object.

        Args
        ----
        file_path: str
            A file path to an existing input file that will be read into this
            object.
        """

        with open(file_path) as dat_file:
            lines = dat_file.readlines()
        self._read_dat_lines(lines)

    def _read_dat_lines(self, dat_lines):
        """Add an existing input file into this object.

        Args
        ----
        dat_lines: [str]
            A list containing the lines of the input file.
        """

        if (
            len(self.nodes)
            + len(self.elements)
            + len(self.materials)
            + len(self.functions)
            > 0
        ):
            raise RuntimeError("A dat file can only be loaded for an empty mesh!")
        if self._dat_file_loaded:
            raise RuntimeError("It is not possible to import two dat files!")

        # Add the lines to this input file.
        self._add_dat_lines(dat_lines)

        if mpy.import_mesh_full:
            # If the solid mesh is imported as objects, link the relevant data
            # after the import.

            # First link the nodes to the elements and sets.
            for element in self.elements:
                for i in range(len(element.nodes)):
                    element.nodes[i] = self.nodes[element.nodes[i]]
            for geometry_type_list in self.geometry_sets.values():
                for geometry_set in geometry_type_list:
                    geometry_set.replace_indices_with_nodes(self.nodes)

            # Link the boundary conditions.
            for bc_key, bc_list in self.boundary_conditions.items():
                for boundary_condition in bc_list:
                    geom_list = self.geometry_sets[bc_key[1]]
                    geom_index = boundary_condition.geometry_set
                    boundary_condition.geometry_set = geom_list[geom_index]
                    boundary_condition.check()

        self._dat_file_loaded = True

    def _add_dat_lines(self, data, **kwargs):
        """Read lines of string into this object."""

        if isinstance(data, list):
            lines = data
        elif isinstance(data, str):
            lines = data.split("\n")
        else:
            raise TypeError(f"Expected list or string but got {type(data)}")

        # Loop over lines and add individual sections separately.
        section_line = None
        section_data = []
        for line in lines:
            line = line.strip()
            if line.startswith("--"):
                self._add_dat_section(section_line, section_data, **kwargs)
                section_line = line
                section_data = []
            else:
                section_data.append(line)
        self._add_dat_section(section_line, section_data, **kwargs)

    def _add_dat_section(self, section_line, section_data, **kwargs):
        """Add a section to the object.

        Args
        ----
        section_line: string
            A string containing the line with the section header. If this is
            None, the data will be added to self.dat_header
        section_data: list(str)
            A list with strings containing the data for this section.
        """

        # The text until the first section will have no section line.
        if section_line is None:
            if not (len(section_data) == 1 and section_data[0] == ""):
                self.dat_header.extend(section_data)
        else:
            # Extract the name of the section.
            name = section_line.strip()
            start = re.search(r"[^-]", name).start()
            section_name = name[start:]

            def group_input_comments(section_data):
                """Group the section data in relevant input data and comment /
                empty lines.

                The comments at the end of the section are lost, as it
                is not clear where they belong to.
                """

                group_list = []
                temp_header_list = []
                for line in section_data:
                    # Check if the line is relevant input data or not.
                    if line.strip() == "" or line.strip().startswith("//"):
                        temp_header_list.append(line)
                    else:
                        group_list.append([line, temp_header_list])
                        temp_header_list = []
                return group_list

            def add_bc(section_header, section_data_comment):
                """Add boundary conditions to the object."""

                # Get the bc and geometry key
                for key, value in self.boundary_condition_names.items():
                    if value == section_header:
                        (bc_key, geometry_key) = key
                        break

                for item, comments in section_data_comment:
                    if mpy.import_mesh_full:
                        self.boundary_conditions.append(
                            (bc_key, geometry_key),
                            BoundaryConditionBase.from_dat(
                                bc_key, item, comments=comments
                            ),
                        )
                    else:
                        self.dat_boundary_conditions.append(
                            (bc_key, geometry_key),
                            BaseMeshItemString(item, comments=comments),
                        )

            def add_set(section_header, section_data_comment):
                """Add sets of points, lines, surfaces or volumes to the
                object.

                We have to do a check of the set index, as it is
                possible that the existing input file skips sections. If
                a section is skipped a dummy section will be inserted,
                so the final numbering matches the sections again.
                """

                def add_to_set(section_header, dat_list, comments):
                    """Add the data_list to the sets of this object."""
                    for key, value in self.geometry_set_names.items():
                        if value == section_header:
                            geometry_key = key
                            break

                    if mpy.import_mesh_full:
                        self.geometry_sets[geometry_key].append(
                            GeometrySetNodes.from_dat(
                                geometry_key, dat_list, comments=comments
                            )
                        )
                    else:
                        self.dat_geometry_sets[geometry_key].append(
                            BaseMeshItemString(dat_list, comments=comments)
                        )

                if len(section_data_comment) > 0:
                    # Add the individual sets to the object. For that loop
                    # until a new set index is reached.
                    last_index = 1
                    dat_list = []
                    current_comments = []
                    for line, comments in section_data_comment:
                        index_line = int(line.split()[-1])
                        if last_index == index_line:
                            dat_list.append(line)
                        elif index_line > last_index:
                            add_to_set(section_header, dat_list, current_comments)
                            # If indices were skipped, add a dummy section
                            # here, so the final ordering will match the
                            # original one.
                            for skip_index in range(last_index + 1, index_line):
                                add_to_set(
                                    section_header,
                                    [f"// Empty set {skip_index}"],
                                    None,
                                )
                            last_index = index_line
                            dat_list = [line]
                            current_comments = comments
                        else:
                            raise ValueError(
                                "The node set indices must be given in ascending order!"
                            )
                    # Add the last set.
                    add_to_set(section_header, dat_list, current_comments)

            def add_line(self_list, line):
                """Add the line to self_list, and handle comments."""
                self_list.append(BaseMeshItemString(line[0], comments=line[1]))

            # Check if the section contains mesh data that has to be added to
            # specific lists.
            section_data_comment = group_input_comments(section_data)
            if section_name == "MATERIALS":
                for line in section_data_comment:
                    add_line(self.materials, line)
            elif section_name == "NODE COORDS":
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        self.nodes.append(Node.from_dat(line))
                    else:
                        add_line(self.dat_nodes, line)
            elif section_name == "STRUCTURE ELEMENTS":
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        self.elements.append(Element.from_dat(line))
                    else:
                        add_line(self.dat_elements, line)
            elif section_name == "THERMO ELEMENTS":
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        self.elements.append(Element.from_dat(line))
                    else:
                        add_line(self.dat_elements, line)
            elif section_name == "TRANSPORT ELEMENTS":
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        self.elements.append(Element.from_dat(line))
                    else:
                        add_line(self.dat_elements, line)
            elif section_name == "FLUID ELEMENTS":
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        raise NotImplementedError(
                            "Fluid elements in combination with mpy.import_mesh_full == True is "
                            "not yet implemented!"
                        )
                    add_line(self.dat_elements_fluid, line)
            elif section_name.startswith("FUNCT"):
                self.functions.append(BaseMeshItemFull(section_data))
            elif section_name in self.boundary_condition_names.values():
                add_bc(section_name, section_data_comment)
            elif section_name.endswith("TOPOLOGY"):
                add_set(section_name, section_data_comment)
            elif section_name == "STRUCTURE KNOTVECTORS":
                self.add_section(
                    InputSectionMultiKey(section_name, section_data, **kwargs)
                )
            else:
                # Section is not in mesh, i.e. simulation parameters.
                self.add_section(InputSection(section_name, section_data, **kwargs))

    def add(self, *args, **kwargs):
        """Add to this object.

        If the type is not recognized, the child add method is called.
        """

        if len(args) == 1 and isinstance(args[0], InputSection):
            self.add_section(args[0], **kwargs)
        elif len(args) == 1 and isinstance(args[0], str):
            self._add_dat_lines(args[0], **kwargs)
        else:
            super().add(*args, **kwargs)

    def add_section(self, section):
        """Add a section to the object.

        If the section name already exists, it is added to that section.
        """
        if section.name in self.sections.keys():
            self.sections[section.name].merge_section(section)
        else:
            self.sections[section.name] = section

    def delete_section(self, section_name):
        """Delete a section from the dictionary self.sections."""
        if section_name in self.sections.keys():
            del self.sections[section_name]
        else:
            raise Warning(f"Section {section_name} does not exist!")

    def write_input_file(self, file_path, *, nox_xml_file=None, **kwargs):
        """Write the input to a file.

        Args
        ----
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
                    os.path.splitext(os.path.basename(file_path))[0] + ".xml"
                )
            else:
                self._nox_xml_file = nox_xml_file

            # Write the xml file to the disc.
            with open(
                os.path.join(os.path.dirname(file_path), self._nox_xml_file), "w"
            ) as xml_file:
                xml_file.write(self.nox_xml)

        with open(file_path, "w") as input_file:
            for line in self.get_dat_lines(**kwargs):
                input_file.write(line)
                input_file.write("\n")

    def get_dat_lines(
        self,
        *,
        header=True,
        dat_header=True,
        add_script_to_header=True,
        check_nox=True,
    ):
        """Return the lines for the input file for the whole object.

        Args
        ----
        header: bool
            If the header should be exported to the input file files.
        dat_header: bool
            If header lines from the imported dat file should be exported.
        append_script_to_header: bool
            If true, a copy of the executing script will be added to the input
            file. This is only in affect when dat_header==True.
        check_nox: bool
            If this is true, an error will be thrown if no nox file is set.
        """

        # Perform some checks on the mesh.
        if mpy.check_overlapping_elements:
            self.check_overlapping_elements()

        # List that will contain all input lines.
        lines = []

        # Add header to the input file.
        end_text = None

        lines.extend(["// " + line for line in mpy.input_file_meshpy_header])
        if header:
            header_text, end_text = self._get_header(add_script_to_header)
            lines.append(header_text)
        if dat_header:
            lines.extend(self.dat_header)

        # Check if a file has to be created for the NOX xml information.
        if self.nox_xml is not None:
            if self._nox_xml_file is None:
                if check_nox:
                    raise ValueError("NOX xml content is given, but no file defined!")
                nox_xml_name = "NOT_DEFINED"
            else:
                nox_xml_name = self._nox_xml_file
            self.add(
                InputSection(
                    "STRUCT NOX/Status Test",
                    f"XML File = {nox_xml_name}",
                    option_overwrite=True,
                )
            )

        # Export the basic sections in the input file.
        for section in self.sections.values():
            if section.name not in self.skip_sections:
                lines.extend(section.get_dat_lines())

        def set_i_global(data_list):
            """Set i_global in every item of data_list."""

            # A check is performed that every entry in data_list is unique.
            if len(data_list) != len(set(data_list)):
                raise ValueError("Elements in data_list are not unique!")

            # Set the values for i_global.
            for i, item in enumerate(data_list):
                item.i_global = i + 1

        def set_i_global_elements(element_list):
            """Set i_global in every item of element_list."""

            # A check is performed that every entry in element_list is unique.
            if len(element_list) != len(set(element_list)):
                raise ValueError("Elements in element_list are not unique!")

            # Set the values for i_global.
            i = 0
            i_nurbs_patch = 0
            for item in element_list:
                # As a NURBS patch can be defined with more elements, an offset is applied to the
                # rest of the items
                item.i_global = i + 1
                if isinstance(item, NURBSPatch):
                    item.n_nurbs_patch = i_nurbs_patch + 1
                    offset = item.get_number_elements()
                    i += offset
                    i_nurbs_patch += 1
                else:
                    i += 1

        def set_i_global_materials(material_list):
            """Set i_global in every item of the materials list.

            We have to account for materials imported from dat files
            that have a random numbering.
            """

            # A check is performed that every entry in material_list is unique.
            if len(material_list) != len(set(material_list)):
                raise ValueError("Elements in material_list are not unique!")

            # Get the maximum material index in materials imported from a string
            max_material_id = 0
            for material in material_list:
                if isinstance(material, BaseMeshItemString):
                    for dat_line in material.get_dat_lines():
                        if dat_line.startswith("MAT "):
                            max_material_id = max(
                                max_material_id, int(dat_line.split(" ")[1])
                            )

            # Set the material id in all MeshPy materials
            i_material = max_material_id + 1
            for material in material_list:
                if not isinstance(material, BaseMeshItemString):
                    material.i_global = i_material
                    i_material += 1

        # Add sets from couplings and boundary conditions to a temp container.
        self.unlink_nodes()
        mesh_sets = self.get_unique_geometry_sets()

        # Combined lists for string type items and "real" mesh type items
        all_nodes = self.dat_nodes + self.nodes
        all_elements_structure = self.dat_elements + self.elements
        all_elements = self.dat_elements_fluid + all_elements_structure
        all_geometry_sets = GeometrySetContainer()
        all_geometry_sets.extend(self.dat_geometry_sets)
        all_geometry_sets.extend(mesh_sets)
        all_boundary_conditions = BoundaryConditionContainer()
        all_boundary_conditions.extend(self.dat_boundary_conditions)
        all_boundary_conditions.extend(self.boundary_conditions)

        # Assign global indices to all entries.
        set_i_global(all_nodes)
        set_i_global_elements(all_elements)
        set_i_global_materials(self.materials)
        set_i_global(self.functions)
        for value in all_geometry_sets.values():
            # We reset the geometry set index here since in self.get_unique_geometry_sets the
            # geometry sets from the dat file are not included.
            set_i_global(value)
        for value in all_boundary_conditions.values():
            set_i_global(value)

        def get_section_dat(section_name, data_list, header_lines=None):
            """Output a section name and apply the get_dat_line for each list
            item."""

            # do not write section if no content is available
            if len(data_list) == 0:
                return

            lines.append(get_section_string(section_name))
            if header_lines:
                if isinstance(header_lines, list):
                    lines.extend(header_lines)
                elif isinstance(header_lines, str):
                    lines.append(header_lines)
                else:
                    raise TypeError(
                        f"Expected string or list, got {type(header_lines)}"
                    )
            for item in data_list:
                lines.extend(item.get_dat_lines())

        # Add material data to the input file.
        get_section_dat("MATERIALS", self.materials)

        # Add the functions.
        for i, funct in enumerate(self.functions):
            lines.append(get_section_string("FUNCT{}".format(i + 1)))
            lines.extend(funct.get_dat_lines())

        # If there are couplings in the mesh, set the link between the nodes
        # and elements, so the couplings can decide which DOFs they couple,
        # depending on the type of the connected beam element.
        def get_number_of_coupling_conditions(key):
            """Return the number of coupling conditions in the mesh."""
            if (key, mpy.geo.point) in all_boundary_conditions.keys():
                return len(all_boundary_conditions[key, mpy.geo.point])
            else:
                return 0

        if (
            get_number_of_coupling_conditions(mpy.bc.point_coupling)
            + get_number_of_coupling_conditions(mpy.bc.point_coupling_penalty)
            > 0
        ):
            self.set_node_links()

        # Add the boundary conditions.
        for (bc_key, geom_key), bc_list in all_boundary_conditions.items():
            if len(bc_list) > 0:
                section_name = (
                    bc_key
                    if isinstance(bc_key, str)
                    else self.boundary_condition_names[bc_key, geom_key]
                )
                get_section_dat(section_name, bc_list)

        # Add additional element sections (e.g. STRUCTURE KNOTVECTORS)
        # We only need to to this on the "real" elements as the imported ones already have their
        # dat sections.
        element_sections = dict()
        for element in self.elements:
            element.add_element_specific_section(element_sections)

        for section in element_sections.values():
            lines.extend(section.get_dat_lines())

        # Add the geometry sets.
        for geom_key, item in all_geometry_sets.items():
            if len(item) > 0:
                get_section_dat(self.geometry_set_names[geom_key], item)

        # Add the nodes and elements.
        get_section_dat("NODE COORDS", all_nodes)
        get_section_dat("STRUCTURE ELEMENTS", all_elements_structure)
        get_section_dat("FLUID ELEMENTS", self.dat_elements_fluid)

        # Add end text.
        if end_text is not None:
            lines.append(end_text)

        return lines

    def get_string(self, **kwargs):
        """Return the lines of the input file as a joined string."""
        return "\n".join(self.get_dat_lines(**kwargs))

    def __str__(self, **kwargs):
        """Return the string representation of this input file."""
        return self.get_string(**kwargs)

    def _get_header(self, add_script):
        """Return the header for the input file."""

        headers = []
        end_text = None

        # Header containing model information.
        current_time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_header = f"// Date:       {current_time_string}\n"
        if self.description:
            model_header += f"// Description: {self.description}\n"
        headers.append(model_header)

        # Get information about the script.
        script_path = os.path.realpath(sys.argv[0])
        script_git_sha, script_git_date = get_git_data(os.path.dirname(script_path))
        script_header = "// Script used to create input file:\n"
        script_header += f"// path:       {script_path}\n"
        if script_git_sha is not None:
            script_header += (
                f"// git sha:    {script_git_sha}\n// git date:   {script_git_date}\n"
            )
        headers.append(script_header)

        # Header containing meshpy information.
        headers.append(
            "// Input file created with meshpy\n"
            f"// git sha:    {mpy.git_sha}\n"
            f"// git date:   {mpy.git_date}\n"
        )

        # Check if cubitpy is loaded.
        if "cubitpy.cubitpy" in sys.modules.keys():
            # Load cubitpy.
            import cubitpy

            # Get git information about cubitpy.
            cubitpy_git_sha, cubitpy_git_date = get_git_data(
                os.path.dirname(cubitpy.__file__)
            )

            if cubitpy_git_sha is not None:
                # Cubitpy_header.
                headers.append(
                    "// The module cubitpy was loaded\n"
                    f"// git sha:    {cubitpy_git_sha}\n"
                    f"// git date:   {cubitpy_git_date}\n"
                )

        string_line = "// " + "".join(["-" for _i in range(mpy.dat_len_section - 3)])

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
