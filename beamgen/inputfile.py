
import datetime

# get version number of beamgen
from beamgen import __VERSION__


 
class BaciOption(object):
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
        
        if len(args) == 1:
            # set from single string
            string = args[0]
            
            # first check if the line has a comment
            first_comment = string.find('//')
            if not first_comment == -1:
                self.option_comment = string[first_comment+2:]
                string = string[:first_comment]
            
            # split up the remaining string into name and value
            self._set_string_split(string.split())
                
        else:
            # set from multiple parameters
            self.option_name = args[0]
            self.option_value = args[1]
            if option_comment:
                self.option_comment = option_comment
        
    
    def _set_string_split(self, string_split):
        """
        Default method to convert a string split list into object parameters.
        Should be overwriten in special child classes.
        """

        if len(string_split) == 2:
            self.option_name = string_split[0]
            self.option_value = string_split[1]
        else:
            print('Error, input string does not match expected!')
        



BaciOption(1,2222, option_comment=1224)
tmp = BaciOption('asd7_fas     h   //fasdf  asdf')
print(tmp.option_name)


class InputSection(object):
    """
    Represent a single section in the input file
    """
    
    def __init__(self, name, data):
        self.name = name
        self.data = data
    
    
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
        
        return self.data



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
        #self.sections 
        
        # holds all the geometry data needed for beam_mesh elements
        self.geometry = None
        
        self.maintainer = maintainer
        self.description = description
    
    
    def get_dat_lines(self):
        """
        Return a list with the lines of the input file
        """
        
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










