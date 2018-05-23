
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
        
        lines.extend(self.mesh.get_dat_lines(print_set_names=print_set_names, print_all_sets=print_all_sets))
        
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






