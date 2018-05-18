
# python packages
import datetime
import re
from _collections import OrderedDict

# meshgen imports
from meshgen.utility import __VERSION__ # version number of beamgen git
from meshgen.mesh import MeshInput
from meshgen.utility import get_section_string

 
class InputLine(object):
    """
    This class is a single option in a baci input file
    """
    
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
        
    
    def __str__(self, *args, **kwargs):
        string = ''
        if not self.option_name == '':
            string += '{:<35} {}{}'.format(self.option_name, self.option_value_pad, self.option_value)
        if not self.option_comment == '':
            if not self.option_name == '':
                string += ' '
            string += '{}'.format(self.option_comment)
        return string




# 
# 
# 
# class MeshSection(object):
#     """
#     A base class to manage mesh items in the input file
#     """
#     
#     def __init__(self, input_file, *args):
#         """
#         Set the default values and call the set_data function witch is overwritten in the child classes.
#         """
#         
#         self.input_file = input_file
#         if len(args) == 0:
#             self.name = None
#         else:
#             self.name = args[0]
#         self.counter = 0
#         self.data = []
# 
# 
#     def get_dat_lines(self):
#         """
#         Return the default lines for this object 
#         """
#         
#         if self.name:
#             return [''.join(['-' for i in range(80-len(self.name))]) + self.name]
#         else:
#             return []
#     
#     
#     def apply_counter_to_geometry(self):
#         """
#         Overwrite in child classes
#         """
#         pass
# 
# 
# class MeshSectionNodes(MeshSection):
#     """
#     Manage nodes from solid and/or beam elements.
#     """
#     
#     def __init__(self, input_file):
#         MeshSection.__init__(self, input_file, 'NODE COORDS')
#     
#     
#     def add_data(self, data):
#         """
#         Add nodal data with lines containing all the nodal information
#         """
#         
#         # store the data in the object
#         self.data = data
#         
#         # set the number of nodes from the last line in data
#         self.counter = int(data[-1].split()[1])
#     
#     
#     def get_dat_lines(self):
#         """
#         Return the nodal data
#         """
#         
#         lines = MeshSection.get_dat_lines(self)
#         lines.extend(self.data)
#         for node in self.input_file.geometry.nodes:
#             lines.append(node.get_dat_line())
#         
#         return lines
#     
#     
#     def apply_counter_to_geometry(self):
#         """
#         Set the global number of nodes
#         """
#         
#         for i, node in enumerate(self.input_file.geometry.nodes):
#             node.n_global = self.counter + i + 1
# 
# 
# class MeshSectionElements(MeshSection):
#     """
#     Manage nodes from solid and/or beam elements.
#     """
#     
#     def __init__(self, input_file):
#         MeshSection.__init__(self, input_file, 'STRUCTURE ELEMENTS')
#     
#     
#     def add_data(self, data):
#         """
#         Add element data with lines containing all the nodal information
#         """
#         
#         # store the data in the object
#         self.data = data
#         
#         # set the elements of nodes from the last line in data
#         self.counter = int(data[-1].split()[0])
#     
#     
#     def get_dat_lines(self):
#         """
#         Return the nodal data
#         """
#         
#         lines = MeshSection.get_dat_lines(self)
#         lines.extend(self.data)
#         for element in self.input_file.geometry.beams:
#             lines.append(element.get_dat_line())
#         
#         return lines
#     
#     
#     def apply_counter_to_geometry(self):
#         """
#         Set the global number of nodes
#         """
#         
#         for i, element in enumerate(self.input_file.geometry.beams):
#             element.n_global = self.counter + i + 1
# 
# 
# class MeshSectionCouplings(MeshSection):
#     
#     def __init__(self, input_file):
#         MeshSection.__init__(self, input_file, 'DESIGN POINT COUPLING CONDITIONS')
#     
#     
#     def add_data(self, data):
#         
#         # get the counter from the previous items
#         self.counter = int(data[0].split()[1])
#         
#         # store the data in the object
#         self.data = data[1:]
#         
#     
#     def get_dat_lines(self):
#         """
#         Return the nodal data
#         """
#         
#         # get number of coupling conditions
#         n_couple = len(self.input_file.geometry.couplings)
#         
#         lines = MeshSection.get_dat_lines(self)
#         lines.append('DPOINT {}'.format(self.counter + n_couple))
#         lines.extend(self.data)
#         
#         for i, couple in enumerate(self.input_file.geometry.couplings):
#             lines.append(couple.get_dat_line())
#         return lines
#     
#     
#     def apply_counter_to_geometry(self):
#         """
#         Set the global number of nodes
#         """
#         
#         for i, element in enumerate(self.input_file.geometry.beams):
#             element.n_global = self.counter + i + 1
# 
# 
# class MeshSectionSets(MeshSection):
#     """
#     Manage sets.
#     """
#     
#     def add_data(self, data):
#         """
#         Add element data with lines containing all the nodal information
#         """
#         
#         # store the data in the object
#         self.data = data
#         
#         # set the elements of nodes from the last line in data
#         self.counter = int(data[-1].split()[0])
#     
#     
#     def get_dat_lines(self):
#         """
#         Return the nodal data
#         """
#         
#         # get the sets for this item
#         sets = self.get_sets()
#         
#         lines = MeshSection.get_dat_lines(self)
#         
#         # print the names of the sets
#         for i, set in enumerate(sets):
#             if not set.name == '':
#                 lines.append('// Set {} {}'.format(i+1+self.counter, set.name))
#             
#         # print input data from solid
#         lines.extend(self.data)
#         
#         # get current sets
#         for i, set in enumerate(sets):
#             for node in set.nodes:
#                 lines.append('NODE {} {} {}'.format(node.n_global, self.get_input_name(), i+1+self.counter))
#         return lines
#     
#     
#     def get_sets(self):
#         if self.name == 'DNODE-NODE TOPOLOGY':
#             return self.input_file.geometry.point_sets    
#         elif self.name == 'DLINE-NODE TOPOLOGY':
#             return self.input_file.geometry.line_sets
#         elif self.name == 'DSURF-NODE TOPOLOGY':
#             return self.input_file.geometry.surf_sets
#         elif self.name == 'DVOL-NODE TOPOLOGY':
#             return self.input_file.geometry.vol_sets
#     
#     def get_input_name(self):
#         if self.name == 'DNODE-NODE TOPOLOGY':
#             return 'DNODE'
#         elif self.name == 'DLINE-NODE TOPOLOGY':
#             return 'DLINE'
#         elif self.name == 'DSURF-NODE TOPOLOGY':
#             return 'DSURF'
#         elif self.name == 'DVOL-NODE TOPOLOGY':
#             return 'DVOL'
#     
#     def get_n_sets(self):
#         return self.counter + len(self.get_sets())
#     
#     def apply_counter_to_geometry(self):
#         """
#         Set the global number of nodes
#         """
#         
#         for i, element in enumerate(self.input_file.geometry.beams):
#             element.n_global = self.counter + i + 1        
# 
# 
# class MeshSectionDesignDescription(MeshSection):
#     def __init__(self, input_file):
#         MeshSection.__init__(self, input_file, 'DESIGN DESCRIPTION')
#         self.n_point_sets = 0
#         self.n_line_sets = 0
#         self.n_surf_sets = 0
#         self.n_vol_sets = 0
#         
#             
#     def get_dat_lines(self):    
#         lines = MeshSection.get_dat_lines(self)
#         lines.append('{:<20} {}'.format('NDPOINT', self.n_point_sets))
#         lines.append('{:<20} {}'.format('NDLINE', self.n_line_sets))
#         lines.append('{:<20} {}'.format('NDSURF', self.n_surf_sets))
#         lines.append('{:<20} {}'.format('NDVOL', self.n_vol_sets))
#         return lines
#     
#     def apply_counter_to_geometry(self):
#         self.n_point_sets = self.input_file.mesh_sections['DNODE-NODE TOPOLOGY'].get_n_sets()
#         self.n_line_sets = self.input_file.mesh_sections['DLINE-NODE TOPOLOGY'].get_n_sets()
#         self.n_surf_sets = self.input_file.mesh_sections['DSURF-NODE TOPOLOGY'].get_n_sets()
#         self.n_vol_sets = self.input_file.mesh_sections['DVOL-NODE TOPOLOGY'].get_n_sets()
#         

        



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



class InputFile(object):
    """ A item that represents a single baci input file. """
    
    def __init__(self, maintainer = '', description = None):
        """ Initialize the main variables. """
        
        # data for header
        self.maintainer = maintainer
        self.description = description
        
        # dictionary for all sections other than mesh sections
        self.sections = OrderedDict()
        
        # set mesh object
        self.mesh = MeshInput()
        
    
    def read_dat(self, file_path):
        """ Read a .dat input file and add the content to this object. """
        
        # empty temp variables
        section_line = None
        section_data = []
                
        with open(file_path) as dat_file:
            for line in dat_file:
                line = line.rstrip()
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
        
        # first text will have no section_line
        if section_line == None:
            # TODO
            pass
        else:
            # extract the name of the section
            name = section_line.strip()
            start = re.search(r'[^-]', name).start()
            section_name = name[start:]
            
            # add to the mesh and check if it should be part of the mesh or before or after
            section_return_value = self.mesh._add_dat_section(section_name, section_data) 
            if section_return_value == 1:
                # add to section to the first section dictionary self
                self.add_section(InputSection(section_name, data=section_data))

    
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

    
    def get_dat_lines(self, header=True):
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
        
        lines.extend(self.mesh.get_dat_lines())
        
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
        
        string = '// Input file created with beamgen git sha: {}\n'.format(__VERSION__)
        string += '// Maintainer: {}\n'.format(self.maintainer)
        string += '// Date: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.description:
            string += '\n// Description: {}'.format(self.description)
        return string

    
    def add_mesh(self, mesh, *args, **kwargs):
        """ Merge the mesh item of the input file with another mesh. """
        
        self.mesh.add_mesh(mesh, *args, **kwargs)






