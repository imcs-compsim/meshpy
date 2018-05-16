
import numpy as np

from beamgen.rotation import Rotation
from beamgen.geometry import Beam3rHerm2Lin3
from beamgen.beam_mesh import BeamMesh
from beamgen.inputfile import InputFile, InputSection


# 
# 
# # create line
# cantilever = BeamMesh()
# cantilever.add_mesh_line(Beam3rHerm2Lin3, np.array([0,0,0]), np.array([10,0,0]), 10)
# end_beam = BeamMesh()
# end_beam.add_mesh_line(Beam3rHerm2Lin3, np.array([10,0,0]), np.array([15,0,0]), 5)
# end_beam.rotate(Rotation([0,0,1],np.pi/2), [10,0,0])
# 
# cantilever.add_mesh(end_beam)



input2 = InputFile(maintainer='Ivo Steinbrecher', description='asd sakf ajsaf slkf jsalkfja sklfj salkfj saklfj salkff')
# input2.geometry = cantilever
#input2.get_dat_lines()

# print(input2._get_header())
# 
# print('end')
# 
# 
# test = InputSection('sdf',['1','2'])
# tmp = test.get_dat_lines()
# print(tmp)


sec = InputSection("asdf", [1,2,3])
sec2 = InputSection("asdf2", [1,2,3])
input2.add_section(sec)
input2.add_section(sec)
input2.add_section(sec2)

print(input2.sections)
lines = input2.get_dat_lines(header=False)
for line in lines:
    print(line)


