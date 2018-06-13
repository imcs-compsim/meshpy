
import numpy as np

from meshpy import Node, Mesh, Rotation


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


def test_rotation():
    """
    Test the performance of rotating nodes
    """
    
    mesh = Mesh()
    create_nodes_array(mesh, 40, 200, 10)
    
    print('number of nodes: {}'.format(len(mesh.nodes)))
    
    start = time.time()
    rot = Rotation([1,1,0],-2)
    mesh.rotate(rot, origin=np.array([1,0,1]))
    end = time.time()
    print('time for rotation: {:7.4f} sec'.format(end - start))
    
    print('finished')


test_rotation()