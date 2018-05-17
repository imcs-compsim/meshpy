
import numpy as np

from beamgen.rotation import Rotation
from beamgen.geometry import Beam3rHerm2Lin3
from beamgen.beam_mesh import BeamMesh
from beamgen.inputfile import InputFile, InputSection, InputSectionNodes, BaciInputLine


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

# 
# 
# input2 = InputFile(maintainer='Ivo Steinbrecher', description='asd sakf ajsaf slkf jsalkfja sklfj salkfj saklfj salkff')
# # input2.geometry = cantilever
# #input2.get_dat_lines()
# 
# # print(input2._get_header())
# # 
# # print('end')
# # 
# # 
# # test = InputSection('sdf',['1','2'])
# # tmp = test.get_dat_lines()
# # print(tmp)
# 
# 
# sec = InputSection("STRUCTURAL DYNAMIC",
#                    """INT_STRATEGY                    Standard""")
# sec2 = InputSectionNodes()
# input2.add_section(sec)
# input2.add_section(sec)
# input2.add_section(sec2)
# 
# cantilever = BeamMesh()
# cantilever.add_mesh_line(Beam3rHerm2Lin3, np.array([0,0,0]), np.array([10,0,0]), 3)
# input2.geometry = cantilever
# 
# print(input2.sections)
# print(input2.get_string(header=False))
# 
# 
# 
# 
# a = BaciLine('test')
# print(str(a))
# 
# 
# 
# 
# print('end')



def line_test():
    """
    Test the BaciLine object
    """
    
    # only test
    tmp = BaciInputLine('text')
    print(tmp)
    print()
    
    # only comment
    tmp = BaciInputLine('// comment')
    print(tmp)
    print()
    
    # text with comment
    tmp = BaciInputLine('var // comment')
    print(tmp)
    print()
    
    # var with value
    tmp = BaciInputLine('var val    // comment')
    print(tmp)
    print()
    
    # no newline
    tmp = BaciInputLine('var \n val comment')
    print(tmp)
    print()
    
    # get with var an val
    tmp = BaciInputLine('var', 'val')
    print(tmp)
    print()
    
    # get with var an val with comment
    tmp = BaciInputLine('var', 'val', option_comment='comment')
    print(tmp)
    print()
    
    # if there is an equal sign the string is split there
    tmp = BaciInputLine('var = val')
    print(tmp)
    print()
    


def test_section():
    """
    Create some test sections
    """

    sec = InputSection("STRUCTURAL DYNAMIC","""INT_STRATEGY                    Standard
    test 2 //test
    test 4    """
    )
    for line in sec.get_dat_lines():
        print(line)
    print()

    sec = InputSection("STRUCTURAL DYNAMIC","""INT_STRATEGY                    Standard
    test 2 //test
    test 4    """,option_overwrite=True)
    sec.add_option('test', 100,option_overwrite=True)
    for line in sec.get_dat_lines():
        print(line)
    print()


def test_input():
    """
    Create a sample input file
    """

    input = InputFile(maintainer='Joe Doe', description='Simple input file')
    sec = InputSection('PROBLEM SIZE', 'DIM 3')
    input.add_section(sec)
    
    sec = InputSection('PROBLEM TYP')
    sec.add_option('PROBLEMTYP', 'Structure')
    sec.add_option('RESTART', '0')
    input.add_section(sec)
    
    sec = InputSection('IO', '''OUTPUT_BIN                      Yes
STRUCT_DISP                     Yes
FILESTEPS                       1000''')
    input.add_section(sec)
    
    sec = InputSection('IO/RUNTIME VTK OUTPUT', '''OUTPUT_DATA_FORMAT              binary
INTERVAL_STEPS                  1
EVERY_ITERATION                 No''')
    input.add_section(sec)
    
    sec = InputSection('IO/RUNTIME VTK OUTPUT/BEAMS',
'''
OUTPUT_BEAMS                    Yes
DISPLACEMENT                    Yes
USE_ABSOLUTE_POSITIONS          Yes
TRIAD_VISUALIZATIONPOINT        Yes
STRAINS_GAUSSPOINT              Yes
INTERNAL_ENERGY_ELEMENT         Yes
''')
    input.add_section(sec)
    
    sec = InputSection(
        'STRUCTURAL DYNAMIC',
        '''
        INT_STRATEGY                    Standard
        LINEAR_SOLVER                   1
        DYNAMICTYP                      Statics
        RESULTSEVRY                     1
        RESEVRYERGY                     1
        RESTARTEVRY                     100
        NLNSOL                          fullnewton
        TIMESTEP                        0.1
        NUMSTEP                         10
        MAXTIME                         1
        PREDICT                         ConstDis
        NORM_DISP                       Abs
        NORM_RESF                       Abs
        NORMCOMBI_RESFDISP              And
        TOLDISP                         1.0E-10
        TOLRES                          1.0E-06
        MAXITER                         100
        '''
        )
    input.add_section(sec)
    
    
    sec = InputSection(
        'STRUCT NOX/Printing',
        '''
        Outer Iteration                 = Yes
        Inner Iteration                 = No
        Outer Iteration StatusTest      = No
        Linear Solver Details           = No
        Test Details                    = No
        Debug                           = NoSTRUCTURAL DYNAMIC
        '''
        )
    input.add_section(sec)
    
    
    sec = InputSection(
        'SOLVER 1',
        '''
        NAME                            Structure_Solver
        SOLVER                          UMFPACK
        '''
        )
    input.add_section(sec)
    
    
    
    print(input.get_string())
    











# line_test()
# test_section()
test_input()

