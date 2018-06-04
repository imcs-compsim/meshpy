
# python packages
import subprocess
import os
import numpy as np


def get_section_string(section_name):
    """ Return the string for a section in the dat file. """
    return ''.join(['-' for i in range(80-len(section_name))]) + section_name


def get_git_sha(repo):
    """
    Return the hash of the last git commit.
    """
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode('ascii').strip()
    return sha


__version__ = '0.0.1'
__git_sha__ = get_git_sha(os.path.dirname(os.path.realpath(__file__)))


def flatten(data):
    """ Flatten out all list items in data. """
    flatten_list = []
    if type(data) == list:
        for item in data:
            flatten_list.extend(flatten(item))
        return flatten_list 
    else:
        return [data]


def _get_close_coordinates(coordinate_list, eps=1e-8):
    """
    Go through coordinates and return a list with all nodes that have the same
    coordinates (within eps). Loop over all coordinates, no division
    """
    
    # number of coordinates
    n_coordinates = len(coordinate_list)
    
    # nodes that have are partner are true
    is_single_coordinate = np.full(n_coordinates, True)
    
    # node indices
    coordinate_indices = np.arange(n_coordinates)
    
    # loop over nodes
    partner_list = []
    for i in range(n_coordinates):
        # check if node already has a partner
        if is_single_coordinate[i]:
            # get distance with all the other nodes and check if the node was already found
            partner_nodes = (np.linalg.norm(coordinate_list - coordinate_list[i],axis=1) < eps) * is_single_coordinate
            # check if there are partner nodes
            if np.sum(partner_nodes) > 1:
                partner_indices = np.extract(partner_nodes, coordinate_indices)
                partner_list.append(partner_indices)
                is_single_coordinate[partner_indices] = False
    
    return partner_list


def get_close_coordinates(coordinate_list, sections=10, eps=1e-8):
    """
    Go through coordinates and return a list with all nodes that have the same
    coordinates (within eps).
    
    The global domain will be divided into n_sections in x-y-z.
    if n_sections is an array with len 3 the segments in x, y and z can be
    given seperately.
    """
    
    # check input
    if type(coordinate_list) == np.ndarray:
        if not (len(np.shape(coordinate_list)) == 2 and np.shape(coordinate_list)[1] == 3):
            print('Error get_close_coordinates, not 2d array')
            return
        n_coordinates = np.shape(coordinate_list)[0]
    else:
        print('Error in get_close_coordinates, coordinates is type {}'.format(type(coordinate_list)))
        return
    
    if type(sections) == int:
        n_sections = np.array([sections for i in range(3)])
    elif type(sections == list) and len(sections) == 3:
        n_sections = np.zeros(3)
        for i, item in enumerate(sections):
            if not(type(item) == int and item > 0):
                print('Error section not as expected!')
            n_sections[i] = item
    else:
        print('Error sections is not of expected type!')
    
    # get the spatial dimensions
    coord_min = np.min(coordinate_list,0)
    dimensions = np.max(coordinate_list,0) - coord_min
    for i in range(3):
        if abs(dimensions[i]) < eps:
            # if dimensions are low in one direction only use one segment there
            n_sections[i] = 1
    
    # modify the dimensions so that integers wont be hit as much
    eps_vector = np.array([eps for i in range(3)])
    coord_min = coord_min - 661*eps_vector
    dimensions = dimensions + 661*eps_vector
        
    # group the coordinates
    section_size = dimensions / n_sections
    current_max = np.array([0.,0.,0.])
    current_min = np.array([0.,0.,0.])
    partner_list = []
    coordinate_indices = np.arange(n_coordinates)
    is_single_coordinate = np.full(n_coordinates, True)
    for ix in range(n_sections[0]):
        for iy in range(n_sections[1]):
            for iz in range(n_sections[2]):
                
                # limits of current section
                current_min = coord_min + [ix,iy,iz]*section_size - eps_vector
                current_max = coord_min + [ix+1,iy+1,iz+1]*section_size + eps_vector
                
                # array with nodes in this section
                diff_max = (current_max - coordinate_list) > 0
                diff_min = (current_min - coordinate_list) < 0
                section_nodes = np.all(diff_min*diff_max,1) * is_single_coordinate
                
                # get matching nodes for this section
                section_indices = np.extract(section_nodes, coordinate_indices)
                partners_local = _get_close_coordinates(np.take(coordinate_list, section_indices, 0), eps=eps)
                
                for item in partners_local:
                    partners_global = np.take(section_indices, item)
                    partner_list.append(list(partners_global))
                    is_single_coordinate[partners_global] = False

    print(len(partner_list))
