import numpy as np
from beamgen.rotation import Rotation
from beamgen.geometry import GeometrySet, Coupling





class BeamMesh(object):
    """
    Holds nodes, beams and couplings of beam_mesh geometry.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Set empty parameters
        """

        self.nodes = []
        self.beams = []
        self.couplings = []
        self.point_sets = []
        self.line_sets = []
        self.surf_sets = []
        self.vol_sets = []
    
    
    def add_mesh(self, beam_mesh, add_sets=True):
        """
        Add other mesh to this one
        """
        
        self.nodes.extend(beam_mesh.nodes)
        self.beams.extend(beam_mesh.beams)
        
#         if add_sets:
#             self.point_sets.extend(beam_mesh.point_sets)
#             self.line_sets.extend(beam_mesh.line_sets)
#             self.mesh_sets.extend(beam_mesh.mesh_sets)
#             self.vol_sets.extend(beam_mesh.vol_sets)
    
    
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
    
    def add_coupling(self, nodes, tmp):
        """
        Add a coupling to the mesh
        """
        
        self.couplings.append(Coupling(nodes, tmp))
    
    
    def add_mesh_line(self, beam_object, start_point, end_point, n, add_sets=True, add_first_node=True):
        """
        A straight line of beam elements.
            n: Number of elements along line
        """
        
        # direction vector of line
        direction = end_point - start_point
        
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
        
        # save the old number of nodes
        node_start = len(self.nodes)
        
        # create the beams
        for i in range(n):
            
            functions = get_beam_function(
                start_point + i*direction/n,
                start_point + (i+1)*direction/n
                )
            
            tmp_beam = beam_object(1)
            if add_first_node and i == 0:
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=True)
            else:
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=False)
            self.beams.append(tmp_beam)
        
        # add nodes to set
        point_sets = [
            GeometrySet('line_point_start', self.nodes[node_start]),
            GeometrySet('line_point_end', self.nodes[-1])
            ]
        line_sets = [
            GeometrySet('line_line', [self.nodes[i] for i in range(node_start,len(self.nodes))])
            ]
        if add_sets:
            self.point_sets.extend(point_sets)
            self.line_sets.extend(line_sets)
        
        return point_sets, line_sets, [], []            






