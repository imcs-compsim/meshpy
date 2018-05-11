
import numpy as np
from rotation import Rotation






class Node(object):
    """
    A class that represents one node in the simulation
    """
    
    def __init__(self, coordinates, rotation=None):
        """
        Each node has a position and an optional rotation object.
        The n_global value is set when the model is writen to a dat file.
        """
        
        self.coordinates = coordinates
        self.rotation = rotation
        self.n_global = None
    
    
    def rotate(self, rotation, only_rotate_triads=False):
        """
        Rotate the node.
        Default values is that the nodes is rotated around the origin.
        If only_rotate_triads is True, then only the triads are rotated,
        the position of the node stays the same.
        """
        
        # apply the roation to the triads
        self.rotation = rotation * self.rotation
        
        # rotate the positions
        if not only_rotate_triads:
            self.coordinates = rotation * self.coordinates


    def get_dat_line(self, n_global):
        """
        Return the line for the dat file for this element.
        """
        
        self.n_global = n_global
        
        return 'NODE {} COORD {} {} {}'.format(
            self.n_global,
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2]
            )


class Beam(object):
    """
    TODO
    """
    
    def __init__(self, material, nodes):
        """
        TODO
        """
        
        self.element_name = None
        if nodes == None:
            self.nodes = []
        else:
            self.nodes = nodes
        self.material = material


class Beam3rHerm2Lin3(Beam):
    """
    TODO
    """
    
    def __init__(self, material, nodes=None):
        """
        Set the data for this beam element
        """
        
        Beam.__init__(self, material, nodes)
        self.element_name = 'BEAM3R HERM2LIN3'
    
    
    def get_dat_line(self, n_element):
        """
        return the line for the dat file for this element
        """
        
        string_nodes = ''
        index = [0, 2, 1]
        for node in [self.nodes[i] for i in index]:
            string_nodes += '{} '.format(node.n_global)
        
        string_triads = ''
        for node in self.nodes:
            string_triads += node.rotation.get_dat()
        
        return '{} {} {}MAT {} TRIADS{} FAD'.format(
            n_element,
            self.element_name,
            string_nodes,
            1,
            string_triads
            )
        
        
    def create_beam(self, nodes, position_function, rotation_function, create_first=False):
        """
        Create the nodes and links for this beam element.
        """
        
        # loop over nodes
        for i, xi in enumerate(np.linspace(-1, 1, 3)):
            # if there is no node in nodes the first one is created no matter what
            # create_first is
            if i > 0 or (len(nodes) == 0 or create_first):
                tmp_node = Node(position_function(xi), rotation=rotation_function(xi))
                nodes.append(tmp_node)
                self.nodes.append(tmp_node)
            else:
                self.nodes.append(nodes[-1])



class BeamMesh(object):
    """
    Holds nodes, beams and couplings of beam_mesh geometry.
    """
    
    def __init__(self, beam_object, *args, **kwargs):
        """
        Set empty parameters
        """
        
        self.beam_object = beam_object
        self.nodes = []
        self.beams = []
        self.couplings = []
        self.sets = []
    
    
    def add_mesh(self, beam_mesh):
        """
        Add other mesh to this one
        """
        
        self.nodes.extend(beam_mesh.nodes)
        self.beams.extend(beam_mesh.beams)
    
    
    def translate(self, vector):
        """
        Move all nodes of this mesh by the vector.
        """
        
        for node in self.nodes:
            node.coordinates += vector
    
    
    def rotate(self, rotation, origin=None):
        """
        Rotate the geometry about the origin.
        """
        
        # move structure to rotation origin
        if origin:
            self.translate(-np.array(origin))
        
        # rotate structure
        for node in self.nodes:
            node.rotate(rotation, only_rotate_triads=False)
        
        # move origin back to initial place
        if origin:
            self.translate(np.array(origin))
    
    
    def wrap_cylinder(self):
        """
        Wrap the geometry around a cylinder.
        The y-z plane morphs into the roation axis.
        There should be NO points with negative x coordinates.
        """
        
        # check the y coordiantes
        for node in self.nodes:
            if node.coordinates[0] < 0:
                print('ERROR, there should be no points with negative x coordiantes')
        
        # transform the nodes
        for node in self.nodes:
            
            # get the cylindercoordinates
            r = np.dot([1,0,0], node.coordinates)
            phi = node.coordinates[1] / r

            # first apply the rotations
            node.rotate(Rotation([0,0,1], phi), only_rotate_triads=True)
            
            # set the new coordinates
            node.coordinates = [
                r * np.cos(phi),
                r * np.sin(phi),
                node.coordinates[2]
                ]
            

class BeamMeshLine(BeamMesh):
    """
    A straight line of beam_mesh elements.
    """
    
    def __init__(self, beam_object, start_point, end_point, n, create_mesh = True, *args, **kwargs):
        """
        Line goes from start_point to end_point with n equidistant elements
        """
        
        BeamMesh.__init__(self, beam_object, *args, **kwargs)
        
        self.start_point = start_point
        self.end_point = end_point
        self.n = n
        
        if create_mesh:
            self.create_mesh()
            
    
    def create_mesh(self):
        """
        Create the mesh for this object.
        """
        
        # direction vector of line
        direction = self.end_point - self.start_point
        
        # rotation for this line (is constant on the whole line)
        t1 = direction
        # check if the x or y axis are larger projected onto the direction
        if np.dot(t1,[1,0,0]) < np.dot(t1,[0,1,0]):
            t2 = [1,0,0]
        else:
            t2 = [0,1,0]
        rotation = Rotation.from_basis(t1, t2)
        
        # this function returns the position and the triads for each element
        def get_beam_function(point_a, point_b):
            
            def position_function(xi):
                return 1/2*(1-xi)*point_a + 1/2*(1+xi)*point_b
            
            def rotation_function(xi):
                return rotation
            
            return (
                position_function,
                rotation_function
                )
        
        # create the beams
        for i in range(self.n):
            
            functions = get_beam_function(
                self.start_point + i*direction/self.n,
                self.start_point + (i+1)*direction/self.n
                )
            
            tmp_beam = self.beam_object(1)
            tmp_beam.create_beam(self.nodes, functions[0], functions[1])
            self.beams.append(tmp_beam)
        

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







# create line
cantilever = BeamMeshLine(Beam3rHerm2Lin3, np.array([0,0,0]), np.array([10,0,0]), 10)
end_beam = BeamMeshLine(Beam3rHerm2Lin3, np.array([10,0,0]), np.array([15,0,0]), 5)
end_beam.rotate(Rotation([0,0,1],np.pi/2), [10,0,0])

cantilever.add_mesh(end_beam)



input2 = InputFile()
input2.geometry = cantilever
input2.get_dat_lines()



print('end')












