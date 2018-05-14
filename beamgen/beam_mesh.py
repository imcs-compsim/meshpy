import numpy as np
from beamgen.rotation import Rotation





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
    
    
    def add_mesh_line(self):
            

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
            self._create_mesh()
            
    
    def _create_mesh(self):
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