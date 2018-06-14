# -*- coding: utf-8 -*-
"""
This module defines the classes that are used to create an input file for Baci.
"""

# python packages
import datetime
import re
from _collections import OrderedDict

# meshgen imports
from . import mpy, get_section_string,  Mesh, BaseMeshItem


 
class InputLine(object):
    """This class is a single option in a Baci input file."""
    
    def __init__(self, *args, option_comment=None, overwrite=False):
        """
        Set the option object.
            - with a single string
            - with two arguments: option_name, option_value 
                and the optional keyword argument option_comment
        """
        
        self.option_name = ''
        self.option_value = ''
        self.option_comment = ''
        self.option_value_pad= '  '
        self.overwrite = overwrite
        
        for arg in args:
            # there should be no newline in the arguments
            if not str(arg).find('\n') == -1:
                print('Error, no newline in InputLine')
        
        if len(args) == 1:
            # set from single string
            string = args[0]
            
            # first check if the line has a comment
            first_comment = string.find('//')
            if not first_comment == -1:
                self.option_comment = string[first_comment:]
                string = string[:first_comment]
            
            # split up the remaining string into name and value
            split = self._set_string_split(string) 
            if split == 1:
                self.option_comment = args[0]
            elif split == 2:
                print('Error in set string split function!')
            
        else:
            # set from multiple parameters
            self.option_name = args[0]
            self.option_value = args[1]
            if option_comment:
                self.option_comment = '// {}'.format(option_comment)
    
    
    def get_key(self):
        """
        Return a key that will be used in the dictionary storage for this item.
        """
        
        if self.option_name == '':
            return self.option_comment
        else:
            return self.option_name
        

    def _set_string_split(self, string):
        """
        Default method to convert a string split list into object parameters.
        Should be overwriten in special child classes.
        
        If everything is ok the return value is 0.
        
        If the return value is 1, option_name and option_value will be empty,
        and the whole input string will be set to option_comment.
        
        If the return value is 2 an error will be thrown. It is also possible
        to throw this error in this function.
        """
        
        # check if there is an equal sign in the string
        if not string.find('=') == -1:
            string_split = string.split('=')
            string_split = [string.strip() for string in string_split]
            self.option_value_pad = '= '
        else:
            string_split = string.split()
        if len(string_split) == 2:
            self.option_name = string_split[0]
            self.option_value = string_split[1]
            return 0
        else:
            return 1
        
    
    def __str__(self):
        string = ''
        if not self.option_name == '':
            string += '{:<35} {}{}'.format(self.option_name, self.option_value_pad, self.option_value)
        if not self.option_comment == '':
            if not self.option_name == '':
                string += ' '
            string += '{}'.format(self.option_comment)
        return string



class InputSection(object):
    """ Represent a single section in the input file. """
    
    
    def __init__(self, name, data=None, option_overwrite=False):
        """ Initiate section. """
        
        self.name = name
        
        # each line in data will be converted to a baci line
        self.data = OrderedDict()
        if data:
            self.add_option(data, option_overwrite=option_overwrite)
    
    
    def merge_section(self, section):
        """ Merge this section with another. This one is the master. """
        
        for option in section.data.values():
            self._add_data(option)
    
    
    def get_dat_lines(self):
        """ Return the dat lines for this section. """
    
        lines = [get_section_string(self.name)]
        lines.extend([str(line) for line in self.data.values()])
        return lines
    
    
    def _add_data(self, option):
        """ Add a InputLine object to the item. """
        
        if not (option.get_key() in self.data.keys()) or option.overwrite:
            self.data[option.get_key()] = option
        elif option.get_key() == '':
            print('TODO? what should happen here?')
        else:
            print('Error, key {} already set!'.format(option.get_key()))
    
    
    def add_option(self, *args, option_comment=None, option_overwrite=False):
        """
        Add data to the section.
        
        data can be:
            string: the string will be split up into lines and added as BaciLine
            list: each element of the list will be added as BaciLine
            option_name & option_value
        """
        
        if len(args) == 1:
            # check type of argument
            if not type(args[0]) == list:
                split = args[0].split('\n')
            else:
                split = args[0]
            
            # remove the first line if it is empty
            if split[0].strip() == '':
                del split[0]
            # remove the last line if it is empty
            if split[-1].strip() == '':
                del split[-1]
            
            for item in split:
                self._add_data(InputLine(item, option_comment=option_comment, overwrite=option_overwrite))
        else:
            self._add_data(InputLine(*args, option_comment=option_comment, overwrite=option_overwrite))












class InputFile(Mesh):
    """A item that represents a complete Baci input file."""
    
    # Define the names of sections and boundary conditions in the input file.
    geometry_set_names = {
        mpy.point:   'DNODE-NODE TOPOLOGY',
        mpy.line:    'DLINE-NODE TOPOLOGY',
        mpy.surface: 'DSURF-NODE TOPOLOGY',
        mpy.volume:  'DVOL-NODE TOPOLOGY'
    }
    boundary_condition_names = {
        (mpy.dirichlet, mpy.point  ): 'DESIGN POINT DIRICH CONDITIONS',
        (mpy.dirichlet, mpy.line   ): 'DESIGN LINE DIRICH CONDITIONS',
        (mpy.dirichlet, mpy.surface): 'DESIGN SURF DIRICH CONDITIONS',
        (mpy.dirichlet, mpy.volume ): 'DESIGN VOL DIRICH CONDITIONS',
        (mpy.neumann,   mpy.point  ): 'DESIGN POINT DIRICH NEUMANN',
        (mpy.neumann,   mpy.line   ): 'DESIGN LINE DIRICH NEUMANN',
        (mpy.neumann,   mpy.surface): 'DESIGN SURF DIRICH NEUMANN',
        (mpy.neumann,   mpy.volume ): 'DESIGN VOL DIRICH NEUMANN'
    }
    boundary_condition_counter = {
        mpy.point:   'DPOINT',
        mpy.line:    'DLINE',
        mpy.surface: 'DSURF',
        mpy.volume:  'DVOL'
    }

    def __init__(self, maintainer = '', description = None, dat_file=None):
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
        
        # Dictionary for all sections other than mesh sections.
        self.sections = OrderedDict()
        
        if not dat_file is None:
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
        
        with open(file_path) as dat_file:
            lines = []
            for line in dat_file:
                lines.append(line)
        self._add_dat_lines(lines)
    
    
    def _add_dat_lines(self, data):
        """Read lines of string into this object.
        """
        
        if isinstance(data, list):
            lines = data
        elif isinstance(data, str):
            lines = data.split('\n')
        else:
            raise TypeError('Expected list or string but got ' + \
                '{}'.format(type(data)))
        
        # Loop over lines and add individual sections separately.
        section_line = None
        section_data = []
        for line in lines:
            line = line.strip()
            if line.startswith('----------'):
                self._add_dat_section(section_line, section_data)
                section_line = line
                section_data = []
            else:
                section_data.append(line)
        self._add_dat_section(section_line, section_data)
    
    
    def _add_dat_section(self, section_line, section_data):
        """
        The values are first added to the mesh object, if the return code is 1, the
        section does not belong to mesh and is added to the self.sections dictionary.
        """
        
        # The text until the first section will have no section line.
        if section_line == None:
            # TODO
            pass
        else:
            # Extract the name of the section.
            name = section_line.strip()
            start = re.search(r'[^-]', name).start()
            section_name = name[start:]
            
            
            def add_bc(section_header, section_data):
                """Add boundary conditions to the object."""
                for i, item in enumerate(section_data):
                    # The first line is the number of BCs and will be skipped.
                    if i > 0:
                        for key, value in self.boundary_condition_names.items():
                            if value == section_header:
                                (bc_key, geometry_key) = key
                                break
                        self.boundary_conditions[bc_key,geometry_key].append(
                            BaseMeshItem(item)
                            )
            
            def add_set(section_header, section_data):
                """
                Add sets of points, lines, surfaces or volumes to the object.
                """
                
                def add_to_set(section_header, dat_list):
                    """Add the data_list to the sets of this object."""
                    for key, value in self.geometry_set_names.items():
                        if value == section_header:
                            geometry_key = key
                            break
                    self.geometry_sets[geometry_key].append(
                        BaseMeshItem(dat_list)
                        )
                
                if len(section_data) > 0:
                    # Add the individual sets to the object. For that loop until
                    # a new set index is reached.
                    last_index = 1
                    dat_list = []
                    for line in section_data:
                        if last_index == int(line.split()[3]):
                            dat_list.append(line)
                        else:
                            last_index = int(line.split()[3])
                            add_to_set(section_header, dat_list)
                            dat_list = [line]
                    # Add the last set.
                    add_to_set(section_header, dat_list)
                    
            
            # Check if the section contains mesh data that has to be added to
            # specific lists.       
            if section_name == 'MATERIALS':
                for line in section_data:
                    self.materials.append(BaseMeshItem(line))
            elif section_name == 'NODE COORDS':
                for line in section_data:
                    self.nodes.append(BaseMeshItem(line))
            elif section_name == 'STRUCTURE ELEMENTS':
                for line in section_data:
                    self.elements.append(BaseMeshItem(line))
            elif section_name.startswith('FUNCT'):
                self.functions.append(BaseMeshItem(section_data))
            elif section_name.endswith('CONDITIONS'):
                add_bc(section_name, section_data)
            elif section_name.endswith('TOPOLOGY'):
                add_set(section_name, section_data)
            elif section_name == 'DESIGN DESCRIPTION' or \
                    section_name == 'END':
                # Skip those sections as they won't be used!
                pass
            else:
                # Section is not in mesh, i.e. simulation parameters.
                self.add_section(InputSection(section_name, data=section_data))
            
        
        
        
        
        

    def add(self, *args, **kwargs):
        """
        Add an item depending on what it is
        """
        
        if len(args) == 1 and isinstance(args[0], InputSection):
            self.add_section(args[0], **kwargs)
        elif len(args) == 1 and isinstance(args[0], str):
            self._add_dat_lines(args[0])
        else:
            Mesh.add(self, *args, **kwargs)
    
    
    def add_section(self, section):
        """
        Add a section to the object.
        If the section name already exists, it is added to that section.
        """
         
        if section.name in self.sections.keys():
            # section already exists, is merged with existing one
            self.sections[section.name].merge_section(section)
        else:
            # add new section
            self.sections[section.name] = section
    
    
    def delete_section(self, section_name):
        """ Delete a section from the dictionary self.sections. """
        
        if section_name in self.sections.keys():
            del self.sections[section_name]
        else:
            print('Warning, section does not exist!')

    def write_input_file(self, file_path, **kwargs):
        """ Write the input to a file. """
        with open(file_path, 'w') as input_file:
            for line in self.get_dat_lines(**kwargs):
                input_file.write(line)
                input_file.write('\n')
        print('Input File written!')
        
    def get_dat_lines(self, header=True, print_set_names=False, print_all_sets=False):
        """ Return the dat lines from all sections. """
        
        # sections not to export in the dat file
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
            'END'
            ]
        
        lines = []
        if header:
            lines.append(self._get_header())
        
        for section in self.sections.values():
            # only export sections that are not in skip_section
            if not section.name in skip_sections:
                lines.extend(section.get_dat_lines())
        
        """
        Get the lines for the input file that contain the information for
        the mesh.
        """
        
        def set_n_global(data_list):
            """ Set n_global in every item of list. """
            for i, item in enumerate(data_list):
                item.n_global = i + 1
        
        def get_section_dat(section_name, data_list, header_lines=None):
            """
            Output a section name and apply the get_dat_line for each list item.
            """
            
            lines.append(get_section_string(section_name))
            if header_lines:
                if isinstance(header_lines, list):
                    lines.extend(header_lines)
                elif isinstance(header_lines, str):
                    lines.append(header_lines)
                else:
                    print('ERROR, you can either add a list or a string')
            for item in data_list:
                lines.extend(item.get_dat_lines())
        
        
        # first all nodes, elements, sets and couplings are assigned a global value
        set_n_global(self.nodes)
        set_n_global(self.elements)
        set_n_global(self.materials)
        set_n_global(self.functions)
        

        
        # first set referenc counter of sets to False, then add bc and then renumber sets
        for i, coupling in enumerate(self.couplings):
            coupling.n_global = i + 1
#             if coupling.node_set.is_referenced:
#                 print('Error this set can not be referenced by something else')
#                 print(coupling.node_set.nodes[0].coordinates)
#                 print(type(coupling.node_set))
#             else:
#                 coupling.node_set.is_referenced = True
        
        # get ordered list of sets and bcs
        for key in self.boundary_conditions.keys():
            set_n_global(self.boundary_conditions[key])
        
        # get dictionary with sets in this mesh
        mesh_sets = self.geometry_sets.copy()
        for coupling in self.couplings:
            mesh_sets[coupling.node_set.geometry_type].append(coupling.node_set)
        for key in self.boundary_conditions.keys():
            for bc in self.boundary_conditions[key]:
                if not bc.is_dat:
                    mesh_sets[bc.geometry_set.geometry_type].append(bc.geometry_set)
        
        for key in mesh_sets.keys():
            set_n_global(mesh_sets[key])

        # add the material data
        get_section_dat('MATERIALS', self.materials)
        
        # add the functions
        for i, funct in enumerate(self.functions):
            lines.append(get_section_string('FUNCT{}'.format(str(i+1))))
            lines.extend(funct.get_dat_lines())
        
        # add the design descriptions
        lines.append(get_section_string('DESIGN DESCRIPTION'))
        lines.append('NDPOINT {}'.format(len(mesh_sets[mpy.point])))
        lines.append('NDLINE {}'.format(len(mesh_sets[mpy.line])))
        lines.append('NDSURF {}'.format(len(mesh_sets[mpy.surface])))
        lines.append('NDVOL {}'.format(len(mesh_sets[mpy.volume])))
        
        # add boundary conditions
        for (bc_key, geom_key) in self.boundary_conditions.keys():
            for i, bc in enumerate(self.boundary_conditions[bc_key, geom_key]):
                if i == 0:
                    lines.append(get_section_string(self.boundary_condition_names[bc_key, geom_key]))
                    lines.append('{} {}'.format(self.boundary_condition_counter[geom_key], len(self.boundary_conditions[bc_key, geom_key])))
                lines.extend(bc.get_dat_lines())

        # add the couplings
        lines.append(get_section_string('DESIGN POINT COUPLING CONDITIONS'))
        lines.append('DPOINT {}'.format(len(self.couplings)))
        for coupling in self.couplings:
            lines.extend(coupling.get_dat_lines())
        
        # add the node sets
        for key in mesh_sets.keys():
            for i, node_set in enumerate(mesh_sets[key]):
                if i == 0:
                    lines.append(get_section_string(self.geometry_set_names[key]))
                lines.extend(node_set.get_dat_lines())

        # add the nodal data
        get_section_dat('NODE COORDS', self.nodes)

        # add the element data
        get_section_dat('STRUCTURE ELEMENTS', self.elements)
        
        
        # the last section is END
        lines.extend(InputSection('END').get_dat_lines())
        
        return lines
    
    
    def get_string(self, header=True):
        """ Return the lines of the input file as string. """
        
        string = ''
        for line in self.get_dat_lines(header=header):
            string += str(line) + '\n'
        return string

    
    def _get_header(self):
        """ Return the header for the input file. """
        
        string = '// Input file created with meshgen git sha: {}\n'.format(mpy.git_sha)
        string += '// Maintainer: {}\n'.format(self.maintainer)
        string += '// Date: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.description:
            string += '\n// Description: {}'.format(self.description)
        return string


