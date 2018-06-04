
# python packages
import numpy as np

# meshgen imports
from meshgen.inputfile import InputFile, InputSection
from meshgen.mesh import Mesh, Beam3rHerm2Lin3, Material, ContainerGeom, BC,\
    GeometrySet, Node, Function, Coupling, __LINE__
from meshgen.rotation import Rotation




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
    tmp = InputLine('text')
    print(tmp)
    print()
    
    # only comment
    tmp = InputLine('// comment')
    print(tmp)
    print()
    
    # text with comment
    tmp = InputLine('var // comment')
    print(tmp)
    print()
    
    # var with value
    tmp = InputLine('var val    // comment')
    print(tmp)
    print()
    
    # no newline
    tmp = InputLine('var \n val comment')
    print(tmp)
    print()
    
    # get with var an val
    tmp = InputLine('var', 'val')
    print(tmp)
    print()
    
    # get with var an val with comment
    tmp = InputLine('var', 'val', option_comment='comment')
    print(tmp)
    print()
    
    # if there is an equal sign the string is split there
    tmp = InputLine('var = val')
    print(tmp)
    print()


def test_section():
    """
    Create some test sections
    """

    sec = InputSection("STRUCTURAL DYNAMIC", """INT_STRATEGY                    Standard
    test 2 //test
    test 4    """
    )
    for line in sec.get_dat_lines():
        print(line)
    print()

    sec = InputSection("STRUCTURAL DYNAMIC", """INT_STRATEGY                    Standard
    test 2 //test
    test 4    """, option_overwrite=True)
    sec.add_option('test', 100, option_overwrite=True)
    for line in sec.get_dat_lines():
        print(line)
    print()


def test_input():
    """
    Create a sample input file
    """

    # create input file
    input = InputFile(maintainer='Joe Doe', description='Simple input file')
    
    # add section with string
    input.add_section_by_data('PROBLEM SIZE', 'DIM 3')
    
    # add section with long string
    input.add_section_by_data(
        'IO/RUNTIME VTK OUTPUT/BEAMS',
        '''
        OUTPUT_BEAMS                    Yes
        DISPLACEMENT                    Yes
        USE_ABSOLUTE_POSITIONS          Yes
        TRIAD_VISUALIZATIONPOINT        Yes
        STRAINS_GAUSSPOINT              Yes
        INTERNAL_ENERGY_ELEMENT         Yes
        ''')
    
    # add section as object
    sec = InputSection('PROBLEM TYP')
    sec.add_option('PROBLEMTYP', 'Structure')
    sec.add_option('RESTART', '0')
    input.add_section(sec)
    
    # add section with equal arguments    
    input.add_section(InputSection(
        'STRUCT NOX/Printing',
        '''
        Outer Iteration                 = Yes
        Inner Iteration                 = No
        Outer Iteration StatusTest      = No
        Linear Solver Details           = No
        Test Details                    = No
        Debug                           = No
        '''
        ))
    
    # delete section
    input.delete_section('IO/RUNTIME VTK OUTPUT/BEAMS')
    input.delete_section('IO/RUNTIME VTK OUTPUT/BEAMS')
    
    # add to section
    input.add_section(InputSection(
        'STRUCT NOX/Printing',
        '''
        Outer Iteration StatusTest      = Yes // this value is overwriten
        Test                            = Maybe // this value is added
        ''', option_overwrite=True))
    input.add_section_by_data(
        'STRUCT NOX/Printing',
        '''
        Outer Iteration StatusTest      = No // this value is overwriten again
        Test                            = Sure // this value is added and overwritten
        ''', option_overwrite=True)

    # add node section
    input.add_section_by_data('NODE COORDS', [
        'NODE 394 COORD 2.00000000e+00 1.00000000e+00 -2.00000003e-01',
        'NODE 395 COORD 2.00000000e+00 1.00000000e+00 -6.00000024e-01',
        'NODE 396 COORD 2.00000000e+00 1.00000000e+00 -1.00000000e+00'])
    
    # add element section
    input.add_section_by_data('STRUCTURE ELEMENTS', [
'1 BEAM3R HERM2LIN3 1 3 2 MAT 1 TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 FAD',
'2 BEAM3R HERM2LIN3 3 5 4 MAT 1 TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 FAD',
'3 BEAM3R HERM2LIN3 5 7 6 MAT 1 TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 FAD',
'4 BEAM3R HERM2LIN3 7 9 8 MAT 1 TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 FAD',
'5 BEAM3R HERM2LIN3 9 11 10 MAT 1 TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 FAD',
'6 BEAM3R HERM2LIN3 11 13 12 MAT 1 TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 FAD'])
    
    # add coupling section
    input.add_section_by_data('DESIGN POINT COUPLING CONDITIONS', [
'DPOINT 100',
'E 5 - NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0',
'E 6 - NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0',
'E 7 - NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0',
'E 8 - NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0'
])

    # add mesh
    cantilever = Mesh()
    nodes1, tmp, tmp, tmp = cantilever.add_beam_mesh_line(Beam3rHerm2Lin3, np.array([0, 0, 0]), np.array([10, 2, 5]), 3000)
    nodes2, tmp, tmp, tmp = cantilever.add_beam_mesh_line(Beam3rHerm2Lin3, np.array([0, 0, 0]), np.array([9, 1, 4]), 3000)
    cantilever.add_coupling([nodes1[0], nodes2[0]], 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0')
    input.mesh = cantilever
    
    print(input.get_string())


def test_sets():
    cantilever = Mesh()
    cantilever.add_beam_mesh_line(Beam3rHerm2Lin3, np.array([0, 0, 0]), np.array([10, 2, 5]), 3000, add_sets=False)
    print(cantilever)
    print(cantilever.point_sets)
    print(cantilever.line_sets)





































def beam_and_solid_tube():
    
    # create input file
    input_file = InputFile(maintainer='My Name', description='Simple input file')
    
    # load solid mesh
    input_file.read_dat('solid-mesh/tube.dat')
    
    # delete solver 2 section
    input_file.delete_section('TITLE')
    
    # add options for beam_output
    input_file.add_section(InputSection(
        'IO/RUNTIME VTK OUTPUT/BEAMS',
        '''
        OUTPUT_BEAMS                    Yes
        DISPLACEMENT                    Yes
        USE_ABSOLUTE_POSITIONS          Yes
        TRIAD_VISUALIZATIONPOINT        Yes
        STRAINS_GAUSSPOINT              Yes
        '''))
    
    # add a straight line beam
    material = Material('MAT_BeamReissnerElastHyper', 1e9, 0, 1e-3, 0.5)
    cantilever = Mesh(name='cantilever')
    cantilever_set = cantilever.add_beam_mesh_line(Beam3rHerm2Lin3, material, [2,0,-5], [2,0,5], 3)
    
    # add fix at start of the beam
    cantilever.add_bc(
        'dirich',
        BC(
            cantilever_set.point[0], # bc set
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0' # bc string
            )
        )
    
    # add displacement controlled bc at end of the beam
    sin = Function('COMPONENT 0 FUNCTION sin(t*2*pi)')
    cos = Function('COMPONENT 0 FUNCTION cos(t*2*pi)')
    cantilever.add_function(sin)
    cantilever.add_function(cos)
    cantilever.add_bc(
        'dirich',
        BC(
            cantilever_set.point[1], # bc set
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 FUNCT {} {} 0 0 0 0 0 0 0', # bc string
            format_replacement=[cos,sin]
            )
        )
    
    # add the beam mesh to the solid mesh
    input_file.add_mesh(cantilever)
    
    # add test case result description
    input_file.add_section(InputSection(
        'RESULT DESCRIPTION',
        '''
        STRUCTURE DIS structure NODE 35 QUANTITY dispx VALUE 1.50796091342925e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 35 QUANTITY dispy VALUE 1.31453288915877e-08 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 35 QUANTITY dispz VALUE 0.0439008100184687e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 69 QUANTITY dispx VALUE 0.921450108160878 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 69 QUANTITY dispy VALUE 1.41113401669104e-15 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 69 QUANTITY dispz VALUE 0.0178350143764099 TOLERANCE 1e-10
        '''))
        
    # write input file
    input_file.write_input_file('/home/ivo/dev/inputgenerator-py/input/meshgen.dat')







def couplings_test():
    
    # create input file
    input_file = InputFile(maintainer='My Name', description='Simple input file')
    
    input_file.add_section(InputSection('PROBLEM SIZE', 'DIM 3'))
    input_file.add_section(InputSection(
        'PROBLEM TYP',
        '''
        PROBLEMTYP                            Structure
        RESTART                               0
        '''))
    input_file.add_section(InputSection(
        'IO',
        '''
        OUTPUT_BIN                            No
        STRUCT_DISP                           No
        FILESTEPS                             1000
        VERBOSITY                             Standard
        '''))
    input_file.add_section(InputSection(
        'IO/RUNTIME VTK OUTPUT',
        '''
        OUTPUT_DATA_FORMAT                    binary
        INTERVAL_STEPS                        1
        EVERY_ITERATION                       No
        '''))
    input_file.add_section(InputSection(
        'STRUCTURAL DYNAMIC',
        '''
        LINEAR_SOLVER                         1
        INT_STRATEGY                          Standard
        DYNAMICTYP                            Statics
        RESULTSEVRY                           1
        RESTARTEVRY                           5
        NLNSOL                                fullnewton
        PREDICT                               TangDis
        TIMESTEP                              0.05
        NUMSTEP                               20
        MAXTIME                               1.0
        TOLRES                                1.0E-5
        TOLDISP                               1.0E-11
        NORM_RESF                             Abs
        NORM_DISP                             Abs
        NORMCOMBI_RESFDISP                    And
        MAXITER                               20
        '''))
    input_file.add_section(InputSection(
        'SOLVER 1',
        '''
        NAME                                  Structure_Solver
        SOLVER                                UMFPACK
        '''))
    input_file.add_section(InputSection(
        'IO/RUNTIME VTK OUTPUT/BEAMS',
        '''
        OUTPUT_BEAMS                    Yes
        DISPLACEMENT                    Yes
        USE_ABSOLUTE_POSITIONS          Yes
        TRIAD_VISUALIZATIONPOINT        Yes
        STRAINS_GAUSSPOINT              Yes
        '''))

    input_file.add_section(InputSection(
        'RESULT DESCRIPTION',
        '''
        STRUCTURE DIS structure NODE 7 QUANTITY dispx VALUE  1.93660652858398 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 7 QUANTITY dispy VALUE 2.96577245498969e-15 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 7 QUANTITY dispz VALUE  -0.377519670507509 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 18 QUANTITY dispx VALUE  1.93660652858398 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 18 QUANTITY dispy VALUE 2.96577245498969e-15 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 18 QUANTITY dispz VALUE  -0.377519670507508 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 30 QUANTITY dispx VALUE  2.24575771708225 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 30 QUANTITY dispy VALUE 3.43921508039788e-15 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 30 QUANTITY dispz VALUE  -0.528911069803332 TOLERANCE 1e-10
        '''))
    
    material = Material('MAT_BeamReissnerElastHyper', 1e9, 0, 1e-3, 0.5)
    cantilevers = Mesh(name='cantilever')
    
    # add three beams
    cantilever_set = []
    cantilever_set.append(cantilevers.add_beam_mesh_line(Beam3rHerm2Lin3, material, [0,0,-5], [0,0,5], 5))
    cantilever_set.append(cantilevers.add_beam_mesh_line(Beam3rHerm2Lin3, material, [2,0,-5], [2,0,1], 3))
    cantilever_set.append(cantilevers.add_beam_mesh_line(Beam3rHerm2Lin3, material, [2,0,1], [2,0,5], 2))
    cantilever_set.append(cantilevers.add_beam_mesh_line(Beam3rHerm2Lin3, material, [4,0,-5], [4,0,1], 3))
    cantilever_set.append(cantilevers.add_beam_mesh_line(Beam3rHerm2Lin3, material, [4,0,1], [4,0,5], 2))
    
    # function for boundary conditions
    sin = Function('COMPONENT 0 FUNCTION sin(t*2*pi)')
    cos = Function('COMPONENT 0 FUNCTION cos(t*2*pi)')
    cantilevers.add_function(sin)
    cantilevers.add_function(cos)
    
    # add boundary conditions
    # fix on the start node
    for index in [0,1,3]:
        cantilevers.add_bc('dirich',
            BC(cantilever_set[index].point[0],
               'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0'
            ))
    # displacement on end node
    for index in [0,2,4]:
        cantilevers.add_bc('dirich',
            BC(cantilever_set[index].point[1],
               'NUMDOF 9 ONOFF 1 1 0 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 FUNCT {} {} 0 0 0 0 0 0 0',
               format_replacement=[cos,sin]
            ))

    # add couplings
    cantilevers.add_coupling(
        Coupling([cantilever_set[1].point[1], cantilever_set[2].point[0]], 'fix')
        )
    cantilevers.add_coupling(
        Coupling([cantilever_set[3].point[1], cantilever_set[4].point[0]], 'joint')
        )
    
    # add the beam mesh to the solid mesh
    input_file.add_mesh(cantilevers)
        
    # write input file
    input_file.write_input_file('/home/ivo/dev/inputgenerator-py/input/meshgen2.dat')







def honeycomb():
    
    # create input file
    input_file = InputFile(maintainer='My Name', description='Simple input file')
    
    input_file.add_section(InputSection('PROBLEM SIZE', 'DIM 3'))
    input_file.add_section(InputSection(
        'PROBLEM TYP',
        '''
        PROBLEMTYP                            Structure
        RESTART                               0
        '''))
    input_file.add_section(InputSection(
        'IO',
        '''
        OUTPUT_BIN                            No
        STRUCT_DISP                           No
        FILESTEPS                             1000
        VERBOSITY                             Standard
        '''))
    input_file.add_section(InputSection(
        'IO/RUNTIME VTK OUTPUT',
        '''
        OUTPUT_DATA_FORMAT                    binary
        INTERVAL_STEPS                        1
        EVERY_ITERATION                       No
        '''))
    input_file.add_section(InputSection(
        'STRUCTURAL DYNAMIC',
        '''
        LINEAR_SOLVER                         1
        INT_STRATEGY                          Standard
        DYNAMICTYP                            Statics
        RESULTSEVRY                           1
        RESTARTEVRY                           5
        NLNSOL                                fullnewton
        PREDICT                               TangDis
        TIMESTEP                              0.05
        NUMSTEP                               20
        MAXTIME                               1.0
        TOLRES                                1.0E-4
        TOLDISP                               1.0E-11
        NORM_RESF                             Abs
        NORM_DISP                             Abs
        NORMCOMBI_RESFDISP                    And
        MAXITER                               20
        '''))
    input_file.add_section(InputSection(
        'SOLVER 1',
        '''
        NAME                                  Structure_Solver
        SOLVER                                UMFPACK
        '''))
    input_file.add_section(InputSection(
        'IO/RUNTIME VTK OUTPUT/BEAMS',
        '''
        OUTPUT_BEAMS                    Yes
        DISPLACEMENT                    Yes
        USE_ABSOLUTE_POSITIONS          Yes
        TRIAD_VISUALIZATIONPOINT        Yes
        STRAINS_GAUSSPOINT              Yes
        '''))
    
    material = Material('MAT_BeamReissnerElastHyper', 1e9, 0, 1e-3, 0.025)
    mesh = Mesh(name='mesh')
    
    # create two meshes with honeycomb structure
    mesh_honeycomb = Mesh(name='honeycomb_' + str(1))
#     honeycomb_set = mesh_honeycomb.add_beam_mesh_honeycomb_flat(Beam3rHerm2Lin3, material, 1, 7, 41, 5,
#                                                                 closed_width=True,
#                                                                 closed_height=True
#                                                                 )
    honeycomb_set = mesh_honeycomb.add_beam_mesh_honeycomb(Beam3rHerm2Lin3, material,
                           1,
                           8,
                           41,
                           3)
    
    # BC
    ft = Function('COMPONENT 0 FUNCTION t')
    mesh_honeycomb.add_function(ft)

    mesh_honeycomb.add_bc('dirich',
            BC(honeycomb_set.point[0],
               'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0'
            ))
    mesh_honeycomb.add_bc('dirich',
            BC(honeycomb_set.point[1],
               'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 -5.0 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0',
               format_replacement=[ft]
            ))
    
    # add the honeycomb to mesh
    mesh.add_mesh(mesh_honeycomb)
            
    # add the beam mesh to the solid mesh
    input_file.add_mesh(mesh)
        
    # write input file
    input_file.write_input_file('/home/ivo/dev/inputgenerator-py/input/honeycomb.dat', print_set_names=True, print_all_sets=True)
    
    





#     set_line2 = cantilever.add_beam_mesh_line(Beam3rHerm2Lin3, material, [5,6,10], [1,2,3], 5)
#     cantilever.add_bc('dirich', BC(set_line2[__POINT__][0], 'NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 0.0 0.0 0.0 0.0 0.0 0.0 FUNCT 0 {} {} 0 0 0', format_replacement=[f2, '6666',22]))
#     bc = BC(set_line2[__LINE__][0], 'asdfasdf')
#     cantilever.add_bc('dirich', bc)
# #     cantilever.add_bc('dirich', BC(set_line2[__POINT__][1], 'asdfasdf'))
# #     sets = GeometrySet('start', [Node([0,0,0])])
# #     cantilever.add_bc('dirich', BC('asdfasdf', sets))

    
    #cantilever.add_function(f2)









def honeycomb_as_input():
    """
    Create the same honeycomb mesh as defined in 
    /Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_honeycomb_stretch_r01_circ10.dat
    """
    
    # create input file
    input_file = InputFile(maintainer='My Name', description='Simple input file')
    #input_file.read_dat('solid-mesh/honeycomb.dat')
    input_file.read_dat('solid-mesh/honeycomb.dat')
    
    input_file.add_section(InputSection(
        'STRUCTURAL DYNAMIC',
        '''
        NUMSTEP                         40
        ''',
        option_overwrite=True))
    
    input_file.add_section(InputSection(
        'RESULT DESCRIPTION',
        '''
        STRUCTURE DIS structure NODE 268 QUANTITY dispx VALUE  0.00000000000000000e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 268 QUANTITY dispy VALUE -8.09347205557697258e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 268 QUANTITY dispz VALUE  2.89298034569662965e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 188 QUANTITY dispx VALUE  0.00000000000000000e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 188 QUANTITY dispy VALUE -8.09347205557697258e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 188 QUANTITY dispz VALUE  2.89298034569662965e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 182 QUANTITY dispx VALUE  0.00000000000000000e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 182 QUANTITY dispy VALUE -8.09347205557697258e+00 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 182 QUANTITY dispz VALUE  2.89298034569662965e+00 TOLERANCE 1e-10
        '''
        ))
    
    material = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0, 1e-3, 0.2, shear_correction=1.1)
    mesh = Mesh(name='mesh')
    
    # create two meshes with honeycomb structure
    mesh_honeycomb = Mesh(name='honeycomb_' + str(1))

    honeycomb_set = mesh_honeycomb.add_beam_mesh_honeycomb(Beam3rHerm2Lin3, material,
                           50,
                           10,
                           4,
                           1)
    
    # BC
    ft = Function('COMPONENT 0 FUNCTION a\nVARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0')
    mesh_honeycomb.add_function(ft)

    # create line sets for the lower nodes
    bc_set = ContainerGeom()
    bc_set.append_item(__LINE__, GeometrySet('line', nodes = honeycomb_set.point[0].nodes))
    bc_set.append_item(__LINE__, GeometrySet('line2', nodes = honeycomb_set.point[1].nodes))
    mesh_honeycomb.sets.merge_containers(bc_set)
  
    mesh_honeycomb.add_bc('dirich',
            BC(bc_set.line[0],
               'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0'
            ))
    mesh_honeycomb.add_bc('dirich',
            BC(bc_set.line[1],
               'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0',
               format_replacement=[ft]
            ))
#     
#     mesh_honeycomb.add_bc('dirich',
#             BC(honeycomb_set.point[0],
#                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0'
#             ))
#     mesh_honeycomb.add_bc('dirich',
#             BC(honeycomb_set.point[1],
#                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0',
#                format_replacement=[ft]
#             ))
    
    # add the honeycomb to mesh
    mesh.add_mesh(mesh_honeycomb)
    
    # rotate the mesh so it is in the same direction as the input mesh
    mesh.rotate(Rotation([0,0,1], np.pi/2))
    
    # add the beam mesh to the solid mesh
    input_file.add_mesh(mesh)
        
    # write input file
    input_file.write_input_file('/home/ivo/dev/inputgenerator-py/input/honeycomb-var1.dat', print_set_names=False, print_all_sets=False)







# line_test()
# test_section()
# test_input()
# test_sets()
#beam_and_solid_tube()
#couplings_test()
#honeycomb()
honeycomb_as_input()
