

import numpy as np
from rotation import Rotation





class Beam(object):
    """
    TODO
    """
    
    _consistency_check = {
        # interpolation: [nNodes, rotations true/false]
        }
    
    def __init__(self, beam_type, interpolation_type, nodes, material):
        """
        TODO
        """
        
        self.beam_type = beam_type
        self.interpolation_type = interpolation_type
        self.nodes = nodes
        self.material = material
        #self.check_nodes_consistency()
    
    
#     def has_rotations(self):
#         """
#         True if the element depends on rotations, false otherwise
#         
#         
#     def check_nodes_consistency(self):
#         """
#         TODO
#         """ 
#         
#         if self.interpolation_type in self._consistency_check.keys():
#             if len(self.nodes) == self._consistency_check[self.interpolation_type][0]:
#                 for node in nodes:
#                     if self._consistency_check[self.interpolation_type][1] and (node.rotation == None):
#                         print("Error in check_nodes_consistency, nodes have no rotations")
#                         return   
#                     if not self._consistency_check[self.interpolation_type][1] and not (node.rotation == None):
#                         print("Error in check_nodes_consistency, nodes have rotations")
#                         return    
#             else:
#                 print("Error in check_nodes_consistency, number of nodes is wrong")
#                 return                    
#         else:
#             print("Error in check_nodes_consistency")
#             return




class Beam3r(Beam):
    """
    TODO
    """
    
    _consistency_check = {
        'HERM2LIN3': [3, True]
        }
    
    def __init__(self, interpolation_type, nodes, material):
        Beam.__init__(self, "BEAM3R", interpolation_type, nodes, material)
    
    
    def get_dat_line(self, n_element):
        """
        return the line for the dat file for this element
        """
        
        string_nodes = ''
        for node in self.nodes:
            string_nodes += '{} '.format(node.n_global)
        
        string_triads = ''
        for node in self.nodes:
            string_triads += node.rotation.get_dat()
        
        return '{} {} {} {}MAT {} TRIADS{} FAD'.format(
            n_element,
            self.beam_type,
            self.interpolation_type,
            string_nodes,
            self.material.n_global,
            string_triads
            )


class BeamMesh(object):
    """
    TODO
    """
    
    # types of beam that are implemented in this class
    _available_beam_types = []
    
    def __init__(self, mesh_type, beam_type, interpolation_type):
        """
        TODO
        """
        
        # check if beam type is valid
        if beam_type in self._available_beam_types:
            print('a')
        self.mesh_type = mesh_type
        self.beam_type = beam_type
        self.interpolation_type = interpolation_type
        self.beams = []
        self.nodes = []
        self.materials = []
        self.dnode = []
        self.dlines = []




class BeamMeshLine(BeamMesh):
    """
    TODO
    """
    
    _available_beam_types = ['test']
    
    def __init__(self, beam_type, interpolation_type, start_point, end_point, n_elements, *args, **kwargs):
        """
        TODO
        """
        
        BeamMesh.__init__(self, 'line', beam_type, interpolation_type, *args, **kwargs)
        
        
        # create the gemetry data
        n_nodes = n_elements * 2 + 1
        diff = end_point - start_point
        
        # create roations (constant for line)
        # tangential vector
        t1 = diff
        # check if the x or y axis are larger projected onto the direction
        if np.dot(t1,[1,0,0]) < np.dot(t1,[0,1,0]):
            t2 = [1,0,0]
        else:
            t2 = [0,1,0]
        rotation = Rotation.from_basis(t1, t2)
        
        # first the nodes
        for i in range(n_nodes):
            self.nodes.append(Node(
                start_point + i / ( n_nodes -1 ) * diff,
                rotation = rotation
                ))
            
        # then the elements
        for i in range(n_elements):
            self.beams.append(Beam3r(
                self.interpolation_type,
                [self.nodes[j] for j in [2*i, 2*i+2, 2*i+1]],
                Node(np.array([0.,0.,0.])) # material
                ))


    def get_dat(self):
        
        lines_nodes = []
        lines_beams = []
        
        for i, node in enumerate(self.nodes):
            lines_nodes.append(node.get_dat_line(i+1))
        
        for i, beam in enumerate(self.beams):
            lines_beams.append(beam.get_dat_line(i+1))
        
        for line in lines_nodes:
            print(line)
            
        for line in lines_beams:
            print(line)
# 
# 
# nodes = []
# nodes.append(Node(np.array([0.,0.,0.]), [1, 1, 1]))
# nodes.append(Node(np.array([0.,0.,1.]), [1, 1, 1]))
# nodes.append(Node(np.array([0.,0.,1.]), [1, 1, 1]))
# 
# beam = Beam3r('HERM2LIN3', nodes, Node(np.array([0.,0.,0.])))
# 
# 
# 
# print(beam.get_dat_line(20))


a = BeamMeshLine('BEAM3R', 'HERM2LIN3', np.array([0,0,0]), 10*np.array([1,2,4]), 200)






tmp = a.get_dat()




#print(np.pi/2)
print('end')


a = Rotation([1,0,0], np.pi)
print(a.get_quaternion())
b = Rotation.from_rotation_matrix(a.get_rotation_matrix())
print(b.get_quaternion())




