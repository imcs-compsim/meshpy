
import datetime

# get version number of beamgen
from beamgen import __VERSION__
from cups import Option


 
class BaciInputLine(object):
    """
    This class is a single option in a baci input file
    """
    
    def __init__(self, *args, option_comment=None):
        """
        Set the option object.
            - with a single string
            - with two arguments: option_name, option_value 
                and the optional keyword argument option_comment
        """
        
        self.option_name = ''
        self.option_value = ''
        self.option_comment = ''
        
        for arg in args:
            # there should be no newline in the arguments
            if not arg.find('\n') == -1:
                print('Error, no newline in BaciInputLine')
        
        if len(args) == 1:
            # set from single string
            string = args[0]
            
            # first check if the line has a comment
            first_comment = string.find('//')
            if not first_comment == -1:
                self.option_comment = string[first_comment:]
                string = string[:first_comment]
            
            # split up the remaining string into name and value
            split = self._set_string_split(string.split()) 
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
        

    def _set_string_split(self, string_split):
        """
        Default method to convert a string split list into object parameters.
        Should be overwriten in special child classes.
        
        If everything is ok the return value is 0.
        
        If the return value is 1, option_name and option_value will be empty,
        and the whole input string will be set to option_comment.
        
        If the return value is 2 an error will be thrown. It is also possible
        to throw this error in this function.
        """

        if len(string_split) == 2:
            self.option_name = string_split[0]
            self.option_value = string_split[1]
            return 0
        else:
            return 1
    
    
    def __str__(self, *args, **kwargs):
        string = ''
        if not self.option_name == '':
            string += '{} {}'.format(self.option_name, self.option_value)
        if not self.option_comment == '':
            if not self.option_name == '':
                string += ' '
            string += '{}'.format(self.option_comment)
        return string


class InputSection(object):
    """
    Represent a single section in the input file
    """
    
    def __init__(self, name, data=None, option_overwrite=False):
        self.name = name
        
        # each line in data will be converted to a baci line
        self.data = {}
        if data:
            self.add_option(data, option_overwrite=option_overwrite)
        
        # a link to the input file object
        self.input_file = None
    
    
    def add_section(self, section):
        """
        Add section to this section.
        """
        
        print('yet to implement')
    
    
    def get_dat_lines(self):
        """
        Return the dat lines for this section.
        In the child classes the function _get_dat_lines has to be defined.
        """
        
        string = ''.join(['-' for i in range(80-len(self.name))])
        string += self.name
        lines = [string]
        lines.extend(self._get_dat_lines())
        return lines
    
    
    def _get_dat_lines(self):
        """
        Per default return the data stored in this object.
        """
        
        return [str(line) for line in self.data.values()]
    
    
    def _add_data(self, option, option_overwrite=False):
        """
        Add a BaciInputLine object to the item.
        """
        
        if not (option.get_key() in self.data.keys()) or option_overwrite:
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
            for item in split:
                self._add_data(BaciInputLine(item, option_comment=option_comment), option_overwrite=option_overwrite)
        else:
            self._add_data(BaciInputLine(*args, option_comment=option_comment), option_overwrite=option_overwrite)


class InputSectionNodes(InputSection):
    """
    Section that stores nodes of the mesh.
    """
    
    def __init__(self):
        InputSection.__init__(self, 'NODE COORDS')
    
    def _get_dat_lines(self):
        """
        Per default return the data stored in this object.
        """
        
        index = 1
        lines = []
        
        for node in self.input_file.geometry.nodes:
            lines.append(node.get_dat_line(index))
            index += 1
        
        return lines


class InputFile(object):
    """
    An object that holds all the information needed for a baci input file.
    """
    
    def __init__(self,
                 maintainer = '',
                 description = None
                 ):
        """
        TODO
        """
        
        # holdes the sections of the inputfile
        self.sections = []
        
        # holds the beam geometry
        self.geometry = None
        
        # data for header
        self.maintainer = maintainer
        self.description = description
        
    
    def _get_section_keys(self):
        """
        Returns a list of the names of the sections in this input file.
        """
        
        return [section.name for section in self.sections]
    
    
    def _get_section_index(self, key):
        """
        Return the index of a section in this item.
        """
        
        if key in self._get_section_keys():
            return self._get_section_keys().index(key)
        else:
            return None
    
    
    def get_section(self, key):
        """
        Return section with name=key. Return false if section is not 
        in this object.
        """
        
        index = self._get_section_index(key)
        if not index == None:
            return self.sections[index]
        else:
            return None
    
    
    def add_section(self,
                    section,
                    add_after=False        
                    ):
        """
        Add a section to the object.
        If the section name already exists, it is added to that section.
        
        Optional: add_after:
            set to name of section that the item will be inserted after (if
            section name does not exist already). Set empty to add ad beginning.
        """
        
        # check if section already exists
        section_base = self.get_section(section.name)
        if section_base:
            section_base.add_section(section)
        else:
            # add link to section
            section.input_file = self
                                    
            # add new section to item
            if add_after:
                if add_after == '':
                    index = -1
                else:
                    index = self._get_section_index(add_after)
            else:
                index = None
            
            # add to list
            if not index == None:
                self.sections.insert(index+1, section)
            else:
                self.sections.append(section)
 
    
    def get_dat_lines(self, header=True):
        """
        Return the dat lines from all sections.
        """
        
        lines = []
        if header:
            lines.append(self._get_header())
        
        for section in self.sections:
            lines.extend(section.get_dat_lines())
        
        return lines
    
    
        lines_nodes = []
        lines_beams = []
        
        for i, node in enumerate(self.geometry.nodes):
            lines_nodes.append(node.get_dat_line(i+1))
        
        for i, beam in enumerate(self.geometry.beams):
            lines_beams.append(beam.get_dat_line(i+1))
        
        for line in lines_nodes:
            print(line)
        
        print('--------------------------------------------------------------STRUCTURE ELEMENTS')
            
        for line in lines_beams:
            print(line)
    
    
    def get_string(self, header=False):
        """
        Return the lines of the input file as string.
        """
        
        string = ''
        for line in self.get_dat_lines(header=header):
            string += str(line) + '\n'
        return string

    
    def _get_header(self):
        """
        Return the header for the input file.
        """
        
        string = '// Input file created with beamgen git sha: {}\n'.format(__VERSION__)
        string += '// Maintainer: {}\n'.format(self.maintainer)
        string += '// Date: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.description:
            string += '\n// Description: {}'.format(self.description)
        return string

    






