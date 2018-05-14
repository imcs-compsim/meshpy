
        

class InputFile(object):
    """
    An object that holds all the information needed for a baci input file.
    """
    
    def __init__(self, *args, **kwargs):
        """
        TODO
        """
        
        # holdes the sections of the inputfile
        #self.sections 
        
        # holds all the geometry data needed for beam_mesh elements
        self.geometry = None
    
    
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


