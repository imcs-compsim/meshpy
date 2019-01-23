# -*- coding: utf-8 -*-
"""
This module defines the classes that are used to create an input file for Baci.
"""

# Python modules.
import sys
import os
import datetime
import re
from _collections import OrderedDict

# Meshpy modules.
from . import mpy, Mesh, BaseMeshItem, Node, Element, BoundaryCondition, \
    GeometrySet
from .utility import get_git_data


def get_section_string(section_name):
    """Return the string for a section in the dat file."""
    return ''.join(
        ['-' for _i in range(mpy.dat_len_section - len(section_name))]
        ) + section_name


class InputLine(object):
    """This class is a single option in a Baci input file."""

    def __init__(self, *args, option_comment=None, option_overwrite=False):
        """
        Set a line of the baci input file.

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

        self.option_name = ''
        self.option_value = ''
        self.option_comment = ''
        self.option_value_pad = '  '
        self.overwrite = option_overwrite

        if len(args) == 1:
            # Set from single a string.
            string = args[0]

            # First check if the line has a comment.
            first_comment = string.find('//')
            if not first_comment == -1:
                self.option_comment = string[first_comment:]
                string = string[:first_comment]

            # Check if there is an equal sign in the string.
            if not string.find('=') == -1:
                string_split = [text.strip() for text in string.split('=')]
                self.option_value_pad = '= '
            elif len(string.split()) == 2:
                string_split = [text.strip() for text in string.split()]
            else:
                string_split = ['', '']
                self.option_comment = args[0].strip()
        else:
            string_split = [str(arg).strip() for arg in args]

        if option_comment is not None:
            if self.option_comment == '':
                self.option_comment = '// {}'.format(option_comment)
            else:
                self.option_comment += ' // {}'.format(option_comment)

        # Check if the string_split array has a suitable size.
        if len(string_split) == 2:
            self.option_name = string_split[0]
            self.option_value = string_split[1]
        else:
            raise ValueError('Could not process the input parameters:'
                + '\nargs:\n    {}\noption_comment:\n    {}'.format(
                    args, option_comment
                    ))

    def get_key(self):
        """
        Return a key that will be used in the dictionary storage for this item.
        If the option_comment is empty the identifier of this object will be
        returned, so than more than one empty lines can be in one section.
        """

        if self.option_name == '':
            if self.option_comment == '':
                return str(id(self))
            else:
                return self.option_comment
        else:
            return self.option_name

    def __str__(self):
        """Return the string for this line of the input file."""
        string = ''
        if not self.option_name == '':
            string += '{:<35} {}{}'.format(
                self.option_name, self.option_value_pad, self.option_value
                )
        if not self.option_comment == '':
            if not self.option_name == '':
                string += ' '
            string += '{}'.format(self.option_comment)
        return string


class InputSection(object):
    """Represent a single section in the input file."""

    def __init__(self, name, *args, **kwargs):

        # Section title.
        self.name = name

        # Each input line will be one entry in this dictionary.
        self.data = OrderedDict()

        for arg in args:
            self.add(arg, **kwargs)

    def add(self, data, **kwargs):
        """
        Add data to this section in the form of an InputLine object.

        Args
        ----
        data: str, list(str)
            If data is a string, it will be split up into lines.
            Each line will be added as an InputLine object.
        """

        if isinstance(data, str):
            data_lines = data.split('\n')
        else:
            # Check if data has entries
            if len(data) == 0:
                return
            data_lines = data

        # Remove the first and or last line if it is empty.
        for index in [0, -1]:
            if data_lines[index].strip() == '':
                del(data_lines[index])

        # Add the data lines.
        for item in data_lines:
            self._add_data(InputLine(item, **kwargs))

    def _add_data(self, option):
        """Add a InputLine object to the item."""

        if (not option.get_key() in self.data.keys()) or option.overwrite:
            self.data[option.get_key()] = option
        else:
            raise KeyError('Key {} is already set!'.format(option.get_key()))

    def merge_section(self, section):
        """Merge this section with another. This one is the master."""

        for option in section.data.values():
            self._add_data(option)

    def get_dat_lines(self):
        """Return the lines for this section in the input file."""

        lines = [get_section_string(self.name)]
        lines.extend([str(line) for line in self.data.values()])
        return lines


class InputFile(Mesh):
    """A item that represents a complete Baci input file."""

    # Define the names of sections and boundary conditions in the input file.
    geometry_set_names = {
        mpy.geo.point:   'DNODE-NODE TOPOLOGY',
        mpy.geo.line:    'DLINE-NODE TOPOLOGY',
        mpy.geo.surface: 'DSURF-NODE TOPOLOGY',
        mpy.geo.volume:  'DVOL-NODE TOPOLOGY'
    }
    boundary_condition_names = {
        (mpy.bc.dirichlet, mpy.geo.point  ): 'DESIGN POINT DIRICH CONDITIONS',
        (mpy.bc.dirichlet, mpy.geo.line   ): 'DESIGN LINE DIRICH CONDITIONS',
        (mpy.bc.dirichlet, mpy.geo.surface): 'DESIGN SURF DIRICH CONDITIONS',
        (mpy.bc.dirichlet, mpy.geo.volume ): 'DESIGN VOL DIRICH CONDITIONS',
        (mpy.bc.neumann,   mpy.geo.point  ): 'DESIGN POINT NEUMANN CONDITIONS',
        (mpy.bc.neumann,   mpy.geo.line   ): 'DESIGN LINE NEUMANN CONDITIONS',
        (mpy.bc.neumann,   mpy.geo.surface): 'DESIGN SURF NEUMANN CONDITIONS',
        (mpy.bc.neumann,   mpy.geo.volume ): 'DESIGN VOL NEUMANN CONDITIONS'
    }
    geometry_counter = {
        mpy.geo.point:   'DPOINT',
        mpy.geo.line:    'DLINE',
        mpy.geo.surface: 'DSURF',
        mpy.geo.volume:  'DVOL'
    }

    # Sections that won't be exported to input file.
    skip_sections = [
        'FLUID ELEMENTS',
        'ALE ELEMENTS',
        'LUBRICATION ELEMENTS',
        'TRANSPORT ELEMENTS',
        'TRANSPORT2 ELEMENTS',
        'THERMO ELEMENTS',
        'ACOUSTIC ELEMENTS',
        'CELL ELEMENTS',
        'CELLSCATRA ELEMENTS',
        'ARTERY ELEMENTS',
        'ELECTROMAGNETIC ELEMENTS',
        'END'
    ]

    def __init__(self, maintainer='', description=None, dat_file=None):
        """
        Initialize the input file.

        Args
        ----
        maintainer: str
            Name of person to maintain this input file.
        description: srt
            Will be shown in the input file as description of the system.
        dat_file: str
            A file path to an existing input file that will be read into this
            object.
        """

        Mesh.__init__(self)

        self.maintainer = maintainer
        self.description = description
        self.dat_header = []

        # Dictionary for all sections other than mesh sections.
        self.sections = OrderedDict()

        # Flag if dat file was loaded.
        self._dat_file_loaded = False

        if dat_file is not None:
            self.read_dat(dat_file)

    def read_dat(self, file_path):
        """
        Read an existing input file into this object.

        Args
        ----
        file_path: str
            A file path to an existing input file that will be read into this
            object.
        """

        if (len(self.nodes) + len(self.elements) + len(self.materials)
                + len(self.functions) + len(self.couplings) > 0):
            raise RuntimeError('A dat file can only be loaded for an '
                + 'empty mesh!')
        if self._dat_file_loaded:
            raise RuntimeError('It is not possible to import two dat files!')

        with open(file_path) as dat_file:
            lines = dat_file.readlines()
        self._add_dat_lines(lines)

        if mpy.import_mesh_full:
            # If the solid mesh is imported as objects, link the relevant data
            # after the import.

            # First link the nodes to the elements and sets.
            for element in self.elements:
                for i in range(len(element.nodes)):
                    element.nodes[i] = self.nodes[element.nodes[i]]
            for geometry_type_list in self.geometry_sets.values():
                for geometry_set in geometry_type_list:
                    for i in range(len(geometry_set.nodes)):
                        geometry_set.nodes[i] = self.nodes[
                            geometry_set.nodes[i]]

            # Link the boundary conditions.
            for bc_key, bc_list in self.boundary_conditions.items():
                for boundary_condition in bc_list:
                    geom_list = self.geometry_sets[bc_key[1]]
                    geom_index = boundary_condition.geometry_set
                    boundary_condition.geometry_set = geom_list[geom_index]

        self._dat_file_loaded = True

    def _add_dat_lines(self, data, **kwargs):
        """Read lines of string into this object."""

        if isinstance(data, list):
            lines = data
        elif isinstance(data, str):
            lines = data.split('\n')
        else:
            raise TypeError('Expected list or string but got '
                + '{}'.format(type(data)))

        # Loop over lines and add individual sections separately.
        section_line = None
        section_data = []
        for line in lines:
            line = line.strip()
            if line.startswith('--'):
                self._add_dat_section(section_line, section_data, **kwargs)
                section_line = line
                section_data = []
            else:
                section_data.append(line)
        self._add_dat_section(section_line, section_data, **kwargs)

    def _add_dat_section(self, section_line, section_data, **kwargs):
        """
        Add a section to the object.

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
            if not (len(section_data) == 1 and section_data[0] == ''):
                self.dat_header.extend(section_data)
        else:
            # Extract the name of the section.
            name = section_line.strip()
            start = re.search(r'[^-]', name).start()
            section_name = name[start:]

            def group_input_comments(section_data):
                """
                Group the section data in relevant input data and comment /
                empty lines. The comments at the end of the section are lost,
                as it is not clear where they belong to.
                """

                group_list = []
                temp_header_list = []
                for line in section_data:
                    # Check if the line is relevant input data or not.
                    if line.strip() == '' or line.strip().startswith('//'):
                        temp_header_list.append(line)
                    else:
                        group_list.append([
                            line,
                            temp_header_list
                            ])
                        temp_header_list = []
                return group_list

            def add_bc(section_header, section_data_comment):
                """Add boundary conditions to the object."""
                for i, [item, comments] in enumerate(section_data_comment):
                    # The first line is the number of BCs and will be skipped.
                    if i > 0:

                        # TODO: move this outside the loop.
                        for key, value in \
                                self.boundary_condition_names.items():
                            if value == section_header:
                                (bc_key, geometry_key) = key
                                break

                        if mpy.import_mesh_full:
                            bc = BoundaryCondition.from_dat(bc_key, item,
                                comments=comments)
                        else:
                            bc = BaseMeshItem(item, comments=comments)

                        self.boundary_conditions[bc_key, geometry_key].append(
                            bc)

            def add_set(section_header, section_data_comment):
                """
                Add sets of points, lines, surfaces or volumes to the object.
                """

                def add_to_set(section_header, dat_list, comments):
                    """Add the data_list to the sets of this object."""
                    for key, value in self.geometry_set_names.items():
                        if value == section_header:
                            geometry_key = key
                            break

                    if mpy.import_mesh_full:
                        geometry_set = GeometrySet.from_dat(geometry_key,
                            dat_list, comments=comments)
                    else:
                        geometry_set = BaseMeshItem(dat_list,
                            comments=comments)
                    self.geometry_sets[geometry_key].append(geometry_set)

                if len(section_data_comment) > 0:
                    # Add the individual sets to the object. For that loop
                    # until a new set index is reached.
                    last_index = 1
                    dat_list = []
                    current_comments = []
                    for line, comments in section_data_comment:
                        if last_index == int(line.split()[3]):
                            dat_list.append(line)
                        else:
                            last_index = int(line.split()[3])
                            add_to_set(section_header, dat_list,
                                current_comments)
                            dat_list = [line]
                            current_comments = comments
                    # Add the last set.
                    add_to_set(section_header, dat_list, current_comments)

            def add_line(self_list, line):
                """Add the line to self_list, and handle comments."""
                self_list.append(BaseMeshItem(line[0], comments=line[1]))

            # Check if the section contains mesh data that has to be added to
            # specific lists.
            section_data_comment = group_input_comments(section_data)
            if section_name == 'MATERIALS':
                for line in section_data_comment:
                    add_line(self.materials, line)
            elif section_name == 'NODE COORDS':
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        self.nodes.append(Node.from_dat(line))
                    else:
                        add_line(self.nodes, line)
            elif section_name == 'STRUCTURE ELEMENTS':
                for line in section_data_comment:
                    if mpy.import_mesh_full:
                        self.elements.append(Element.from_dat(line))
                    else:
                        add_line(self.elements, line)
            elif section_name.startswith('FUNCT'):
                self.functions.append(BaseMeshItem(section_data))
            elif section_name.endswith('CONDITIONS'):
                add_bc(section_name, section_data_comment)
            elif section_name.endswith('TOPOLOGY'):
                add_set(section_name, section_data_comment)
            elif section_name == 'DESIGN DESCRIPTION' or \
                    section_name == 'END':
                # Skip those sections as they won't be used!
                pass
            else:
                # Section is not in mesh, i.e. simulation parameters.
                self.add_section(
                    InputSection(section_name, section_data, **kwargs)
                    )

    def add(self, *args, **kwargs):
        """
        Add to this object. If the type is not recognized, the child add method
        is called.
        """

        if len(args) == 1 and isinstance(args[0], InputSection):
            self.add_section(args[0], **kwargs)
        elif len(args) == 1 and isinstance(args[0], str):
            self._add_dat_lines(args[0], **kwargs)
        else:
            Mesh.add(self, *args, **kwargs)

    def add_section(self, section):
        """
        Add a section to the object.
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
            raise Warning('Section {} does not exist!'.format(section_name))

    def write_input_file(self, file_path, **kwargs):
        """Write the input to a file."""
        with open(file_path, 'w') as input_file:
            for line in self.get_dat_lines(**kwargs):
                input_file.write(line)
                input_file.write('\n')

    def get_dat_lines(self, header=True, dat_header=True,
            add_script_to_header=True):
        """
        Return the lines for the input file for the whole object.

        Args
        ----
        header: bool
            If the header should be exported to the input file files.
        dat_header: bool
            If header lines from the imported dat file should be exported.
        append_script_to_header: bool
            If true, a copy of the executing script will be added to the input
            file. This is only in affect when dat_header==True.
        """

        # List that will contain all input lines.
        lines = []

        # Add header to the input file.
        end_text = None
        if header:
            header_text, end_text = self._get_header(add_script_to_header)
            lines.append(header_text)
        if dat_header:
            lines.extend(self.dat_header)

        # Export the basic sections in the input file.
        for section in self.sections.values():
            if section.name not in self.skip_sections:
                lines.extend(section.get_dat_lines())

        def set_n_global(data_list):
            """Set n_global in every item of data_list."""

            # A check is performed that every entry in data_list is unique.
            if len(data_list) != len(set(data_list)):
                raise ValueError('Elements in data_list are not unique!')

            # Set the values for n_global.
            for i, item in enumerate(data_list):
                item.n_global = i + 1

        # Add sets from couplings and boundary conditions to a temp container.
        self.unlink_nodes()
        mesh_sets = self.get_unique_geometry_sets()

        # Assign global indices to all entries.
        set_n_global(self.nodes)
        set_n_global(self.elements)
        set_n_global(self.materials)
        set_n_global(self.functions)
        set_n_global(self.couplings)
        for key in self.boundary_conditions.keys():
            set_n_global(self.boundary_conditions[key])

        def get_section_dat(section_name, data_list, header_lines=None):
            """
            Output a section name and apply the get_dat_line for each list
            item.
            """
            lines.append(get_section_string(section_name))
            if header_lines:
                if isinstance(header_lines, list):
                    lines.extend(header_lines)
                elif isinstance(header_lines, str):
                    lines.append(header_lines)
                else:
                    raise TypeError('Expected string or list, got {}'.format(
                        type(header_lines)))
            for item in data_list:
                lines.extend(item.get_dat_lines())

        # Add material data to the input file.
        get_section_dat('MATERIALS', self.materials)

        # Add the functions.
        for i, funct in enumerate(self.functions):
            lines.append(get_section_string('FUNCT{}'.format(i + 1)))
            lines.extend(funct.get_dat_lines())

        # Add the design description.
        lines.append(get_section_string('DESIGN DESCRIPTION'))
        lines.append('NDPOINT {}'.format(len(mesh_sets[mpy.geo.point])))
        lines.append('NDLINE {}'.format(len(mesh_sets[mpy.geo.line])))
        lines.append('NDSURF {}'.format(len(mesh_sets[mpy.geo.surface])))
        lines.append('NDVOL {}'.format(len(mesh_sets[mpy.geo.volume])))

        # Add the boundary conditions.
        for (bc_key, geom_key), bc_list in self.boundary_conditions.items():
            if len(bc_list) > 0:
                get_section_dat(
                    self.boundary_condition_names[bc_key, geom_key],
                    bc_list,
                    header_lines='{} {}'.format(
                        self.geometry_counter[geom_key],
                        len(self.boundary_conditions[bc_key, geom_key])
                        )
                    )

        # Add the couplings.
        if len(self.couplings) > 0:
            # Set the link for the nodes, so the couplings can decide which
            # DOFs they couple.
            self.set_node_links()

            get_section_dat(
                get_section_string('DESIGN POINT COUPLING CONDITIONS'),
                self.couplings,
                header_lines='DPOINT {}'.format(len(self.couplings))
                )

        # Add the geometry sets.
        for geom_key, item in mesh_sets.items():
            if len(item) > 0:
                get_section_dat(
                    self.geometry_set_names[geom_key],
                    item
                    )

        # Add the nodes and elements.
        get_section_dat('NODE COORDS', self.nodes)
        get_section_dat('STRUCTURE ELEMENTS', self.elements)

        # The last section is END
        lines.extend(InputSection('END').get_dat_lines())

        # Add end text.
        if end_text is not None:
            lines.append(end_text)

        return lines

    def get_string(self, **kwargs):
        """Return the lines of the input file as string."""
        return '\n'.join(self.get_dat_lines(**kwargs))

    def __str__(self, **kwargs):
        return self.get_string(**kwargs)

    def _get_header(self, add_script):
        """Return the header for the input file."""

        headers = []
        end_text = None

        # Header containing model information.
        model_header = (
            '// Maintainer: {}\n' +
            '// Date:       {}\n').format(
                self.maintainer,
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
        if self.description:
            model_header += '// Description: {}\n'.format(self.description)
        headers.append(model_header)

        # Get information about the script.
        script_path = os.path.realpath(sys.argv[0])
        script_git_sha, script_git_date = get_git_data(os.path.dirname(
            script_path))
        script_header = '// Script used to create input file:\n'
        script_header += '// path:       {}\n'.format(script_path)
        if script_git_sha is not None:
            script_header += (
                '// git sha:    {}\n' +
                '// git date:   {}\n').format(script_git_sha, script_git_date)
        headers.append(script_header)

        # Header containing meshpy information.
        headers.append(('// Input file created with meshpy {}\n'
            + '// git sha:    {}\n'
            + '// git date:   {}\n').format(
                mpy.version,
                mpy.git_sha,
                mpy.git_date
                ))

        # Check if cubitpy is loaded.
        if 'cubitpy.cubitpy' in sys.modules.keys():

            # Load cubitpy.
            import cubitpy

            # Get git information about cubitpy.
            cubitpy_git_sha, cubitpy_git_date = get_git_data(
                os.path.dirname(cubitpy.__file__))

            if cubitpy_git_sha is not None:
                # Cubitpy_header.
                headers.append(('// The module cubitpy was loaded\n'
                    + '// git sha:    {}\n'
                    + '// git date:   {}\n').format(
                        cubitpy_git_sha,
                        cubitpy_git_date
                        ))

        string_line = '// ' + ''.join(
            ['-' for _i in range(mpy.dat_len_section - 3)])

        # If needed, append the contents of the script.
        if add_script:
            # Header for the script 'section'.
            script_lines = [string_line
                + '\n// Full script used to create this input file.\n'
                + string_line + '\n']

            # Get the contents of script.
            with open(script_path) as script_file:
                script_lines.extend(script_file.readlines())

            # Comment the python code lines.
            end_text = '//'.join(script_lines)

        return (string_line + '\n' + (string_line + '\n').join(headers) + \
            string_line), end_text

    def set_default_header_static(self, *,
            time_step=1.,
            n_steps=1,
            max_time=None,
            max_iter=20,
            tol_res=1e-7,
            tol_disp=1e-11,
            binning_bounding_box=None,
            option_overwrite=False
            ):
        """
        Set default header parameters for a static analysis.
        """

        # Set values for the parameters that can not directly be set by
        # keyword arguments.
        if binning_bounding_box is None:
            # No binning.
            binning = False
        elif isinstance(binning_bounding_box, bool) and binning_bounding_box:
            # Default binning.
            binning = True
            binning_bounding_box = [-1, -1, -1, 1, 1, 1]
        elif (isinstance(binning_bounding_box, list)
                and len(binning_bounding_box) == 6):
            # User given boundary box.
            binning = True

        self.add('''
            ------------------------------------PROBLEM SIZE
            DIM 3
            ------------------------------------PROBLEM TYP
            PROBLEMTYP                            Structure
            RESTART                               0
            --------------------------------------IO
            OUTPUT_BIN                            No
            STRUCT_DISP                           No
            FILESTEPS                             1000
            VERBOSITY                             Standard
            ''', option_overwrite=option_overwrite)

        # Output / beam output.
        self.add(InputSection(
            'IO/RUNTIME VTK OUTPUT',
            '''
            OUTPUT_DATA_FORMAT                    binary
            INTERVAL_STEPS                        1
            EVERY_ITERATION                       No
            ''', option_overwrite=option_overwrite))
        self.add(InputSection(
            'IO/RUNTIME VTK OUTPUT/BEAMS',
            '''
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            ''', option_overwrite=option_overwrite))

        # Problem type settings.
        if max_time is None:
            max_time = time_step * n_steps
        self.add(InputSection('STRUCTURAL DYNAMIC',
            '''
            LINEAR_SOLVER                         1
            INT_STRATEGY                          Standard
            DYNAMICTYP                            Statics
            RESULTSEVRY                           1
            NLNSOL                                fullnewton
            PREDICT                               TangDis
            TIMESTEP                              {0}
            NUMSTEP                               {1}
            MAXTIME                               {2}
            TOLRES                                {3}
            TOLDISP                               {4}
            NORM_RESF                             Abs
            NORM_DISP                             Abs
            NORMCOMBI_RESFDISP                    And
            MAXITER                               {5}
            '''.format(
                time_step,
                n_steps,
                max_time,
                tol_res,
                tol_disp,
                max_iter
                ),
            option_overwrite=option_overwrite))

        # Solver
        self.add(InputSection(
            'SOLVER 1',
            '''
            NAME                                  Structure_Solver
            SOLVER                                Superlu
            ''', option_overwrite=option_overwrite))

        # Binning strategy.
        if binning:
            bounding_box_string = ''
            for val in binning_bounding_box:
                bounding_box_string += ' {}'.format(val)
            self.add(InputSection(
                'BINNING STRATEGY',
                '''
                CUTOFF_RADIUS 2
                BOUNDINGBOX {}
                '''.format(bounding_box_string),
                option_overwrite=option_overwrite))
