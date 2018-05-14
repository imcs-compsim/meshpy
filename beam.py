
import numpy as np

from beamgen.rotation import Rotation
from beamgen.geometry import Beam3rHerm2Lin3
from beamgen.beam_mesh import BeamMesh, BeamMeshLine
from beamgen.inputfile import InputFile




# create line
cantilever = BeamMeshLine(Beam3rHerm2Lin3, np.array([0,0,0]), np.array([10,0,0]), 10)
end_beam = BeamMeshLine(Beam3rHerm2Lin3, np.array([10,0,0]), np.array([15,0,0]), 5)
end_beam.rotate(Rotation([0,0,1],np.pi/2), [10,0,0])

cantilever.add_mesh(end_beam)



input2 = InputFile()
input2.geometry = cantilever
input2.get_dat_lines()



print('end')


