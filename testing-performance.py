
import numpy as np

from meshpy import *
from meshpy import Node

import time




def create_nodes_array(mesh, n_sec,n_per_sec,l):
    """
    Create an grid of nodes with n_per_sec+1 nodes per section
    """
    
    rot = Rotation([1,1,0],2)
    base_x = np.array([1.,0,0])*l/n_sec
    base_y = np.array([0,1.,0])*l/n_sec
    min_x = np.array([1.,0,0])*l/n_sec/n_per_sec
    min_y = np.array([0,1.,0])*l/n_sec/n_per_sec
    
    for ix in range(n_sec):
        for iy in range(n_sec):
            for inx in range(n_per_sec+1):
                if inx < n_per_sec or ix == n_sec-1:
                    mesh.nodes.append(Node(base_x*ix + base_y*iy + min_x*inx, rot))
            for iny in range(n_per_sec+1):
                if iny < n_per_sec or iy == n_sec-1:
                    mesh.nodes.append(Node(base_x*ix + base_y*iy + min_y*iny, rot))


def create_elements_array(mesh, n_sec,n_per_sec,l):
    """
    Create an grid of nodes with n_per_sec+1 nodes per section
    """
    
    # Add material and functions.
    mat = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0.3, 1e-3, 0.2,
        shear_correction=0.75
        )
    mesh.add(mat)
    
    rot = Rotation([1,1,0],2)
    base_x = np.array([1.,0,0])*l/n_sec
    base_y = np.array([0,1.,0])*l/n_sec
    min_x = np.array([1.,0,0])*l/n_sec/n_per_sec
    min_y = np.array([0,1.,0])*l/n_sec/n_per_sec
    
    for ix in range(n_sec):
        for iy in range(n_sec):
            mesh.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
                base_x*ix + base_y*iy,
                base_x*ix + base_y*(iy+1),
                n_el = n_per_sec)
            mesh.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
                base_x*ix + base_y*iy,
                base_x*(ix+1) + base_y*iy,
                n_el = n_per_sec)


def test_rotation():
    """
    Test the performance of rotating nodes
    """
    
    start = time.time()
    mesh = Mesh()
    #create_nodes_array(mesh, 100, 50, 10)
    #create_nodes_array(mesh, 40, 20, 10)
    create_nodes_array(mesh, 20, 20, 10)
    #create_nodes_array(mesh, 4, 2, 10)
    #create_elements_array(mesh, 100, 50, 10)
    #create_elements_array(mesh, 40, 20, 10)
    #create_elements_array(mesh, 4, 2, 10)
    end = time.time()
    print('time for creation: {:7.4f} sec'.format(end - start))
    
    
    
    print('number of nodes: {}'.format(len(mesh.nodes)))
    
    start = time.time()
    rot = Rotation([1,1,0],-2)
    mesh.rotate(rot, origin=np.array([1,0,1]))
    end = time.time()
    print('time for rotation: {:7.4f} sec'.format(end - start))
    
    start = time.time()
    mesh.wrap_around_cylinder()
    end = time.time()
    print('time for wraping: {:7.4f} sec'.format(end - start))

    start = time.time()
    mesh.add_connections()
    end = time.time()
    print('time for connections: {:7.4f} sec'.format(end - start))
    
    print('finished')


test_rotation()