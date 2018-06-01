import unittest
import numpy as np
import os
import subprocess

# import modules from meshgen
from meshgen.rotation import Rotation
from meshgen.inputfile import InputFile, InputSection
from meshgen.mesh import Material, Mesh, Function, Beam3rHerm2Lin3, ContainerGeom, __LINE__, GeometrySet, BC


from meshgen.utility import get_close_coordinates


import time

start = time.time()
print("hello")
end = time.time()
print(end - start)


print()
print()
print()
print()
print()

"""
Create the same honeycomb mesh as defined in 
/Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_honeycomb_stretch_r01_circ10.dat
"""


# create two meshes with honeycomb structure
mesh_honeycomb = Mesh(name='honeycomb_' + str(1))

honeycomb_set = mesh_honeycomb.add_beam_mesh_honeycomb(Beam3rHerm2Lin3, 1,
                       50,
                       20,
                       80,
                       5)

print(len(mesh_honeycomb.nodes))
# coord = [node.coordinates for node in mesh_honeycomb.nodes]
# 
# 
# coordinates = np.array(coord )
# print(len(coord))
# 
# start = time.time()
# get_close_coordinates(coordinates, sections=20)
# end = time.time()
# print(end - start)
# print('time for new')
# print()