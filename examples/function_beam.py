# -*- coding: utf-8 -*-
"""
Create a beam from a parametric function.
"""

# Python modules
import autograd.numpy as np
#import numpy as np

# Meshpy modules.
from meshpy import *


default_parameters = '''
--------------------------------------------------------------------PROBLEM SIZE
DIM                                   3
---------------------------------------------------------------------PROBLEM TYP
PROBLEMTYP                            Structure
RESTART                               0
------------------------------------------------------------------DISCRETISATION
NUMFLUIDDIS                           0
NUMSTRUCDIS                           1
NUMALEDIS                             0
NUMTHERMDIS                           0
-----------------------------------------------------------IO/RUNTIME VTK OUTPUT
OUTPUT_DATA_FORMAT                    binary
INTERVAL_STEPS                        1
EVERY_ITERATION                       No
-----------------------------------------------------IO/RUNTIME VTK OUTPUT/BEAMS
OUTPUT_BEAMS                          Yes
DISPLACEMENT                          Yes
USE_ABSOLUTE_POSITIONS                Yes
TRIAD_VISUALIZATIONPOINT              Yes
STRAINS_GAUSSPOINT                    Yes
MATERIAL_FORCES_GAUSSPOINT            Yes
--------------------------------------------------------------STRUCTURAL DYNAMIC
INT_STRATEGY                          Standard
LINEAR_SOLVER                         1
DYNAMICTYP                            Statics
RESULTSEVRY                           1
NLNSOL                                fullnewton
TOLRES                                1.0E-7
TOLDISP                               1.0E-10
NORM_RESF                             Abs
NORM_DISP                             Abs
NORMCOMBI_RESFDISP                    And
MAXITER                               15
PREDICT                               TangDis
------------------------------------------------------------------------SOLVER 1
NAME                                  Structure_Solver
SOLVER                                UMFPACK
'''




def compare_helix():
    """
    Create a helix by two different methods:
        1. By wrapping a line in space around a cylinder.
        2. With a parametric curve in space.
    """
    
    input_file = InputFile(maintainer='Ivo Steinbrecher')
    input_file.add(default_parameters)
    input_file.add('''
        ------------------------STRUCTURAL DYNAMIC
        TIMESTEP                              0.1
        NUMSTEP                               10
        MAXTIME                               1.0
        ''')
    
    # Add material and functions.
    mat = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0.3, 1e-3, 0.5,
        shear_correction=0.75
        )
    ft = Function('COMPONENT 0 FUNCTION t')
    input_file.add(mat, ft)
    
    # Set parameters for the helix.
    R = 2.
    tz = 1. # incline
    n = 3 # number of turns
    n_el = 10
    
    # Sets to apply boundary conditions on.
    sets = []
    
    # Create a line and wrap it around a cylinder.
    sets.append(input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [R, 0, 0], [R, 2*np.pi*n*R, n*tz], n_el=n_el))
    input_file.wrap_around_cylinder()
    
    # Create a helix with a parametric curve.
    offset_x = 2.2 * R
    def helix(t):
        return np.array([offset_x + R*np.cos(t), R*np.sin(t), t*tz/(2*np.pi)])
    sets.append(input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat, helix,
        [0,2*np.pi*n], n_el=n_el))
    
    # Apply boundary conditions.
    for node_set in sets:
        input_file.add(BoundaryCondition(
            node_set['start'],
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
            bc_type=mpy.dirichlet
            ))
        input_file.add(BoundaryCondition(
            node_set['end'],
            'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
            format_replacement=[0.01, ft],
            bc_type=mpy.neumann
            ))
    
    # Write input file.
    input_file.write_input_file('/home/ivo/temp/curve_helix.dat')


def compare_lines():
    """
    Create two lines, one with  trivial triad positions, the other one with
    rotating triads.
    """
    
    input_file = InputFile(maintainer='Ivo Steinbrecher')
    input_file.add(default_parameters)
    input_file.add('''
        ------------------------STRUCTURAL DYNAMIC
        TIMESTEP                              0.1
        NUMSTEP                               10
        MAXTIME                               1.0
        ''')
    
    # Add material and functions.
    mat = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0.3, 1e-3, 0.5,
        shear_correction=0.75
        )
    ft = Function('COMPONENT 0 FUNCTION t')
    input_file.add(mat, ft)
    
    # Set parameters for the lines.
    L = 10.
    n_el = 10
    phi_end = 2*np.pi
    
    # Sets to apply boundary conditions on.
    sets = []
    
    # Create a line.
    sets.append(input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [L, 0, 0], [0, 0, 0], n_el=n_el))
    
    # Create a line with a parametric curve.
    offset = 0.2*L
    def line(t):
        return np.array([L - t*L, offset, 0.])
    def rotation(t):
        return Rotation([1,0,0], phi_end * t) * Rotation([0,0,1], np.pi) 
    sets.append(input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat, line,
        [0,1], n_el=n_el, function_rotation=rotation))
    
    # Apply boundary conditions.
    for node_set in sets:
        input_file.add(BoundaryCondition(
            node_set['end'],
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
            bc_type=mpy.dirichlet
            ))
        input_file.add(BoundaryCondition(
            node_set['start'],
            'NUMDOF 9 ONOFF 0 1 0 1 0 0 0 0 0 VAL {0} {2} {0} {0} {0} {0} {0} {0} {0} FUNCT {1} {1} {1} {1} {1} {1} {1} {1} {1}',
            format_replacement=[-0.1, ft, 0.05],
            bc_type=mpy.neumann
            ))
        
    input_file.add(InputSection(
        'RESULT DESCRIPTION',
        '''
        STRUCTURE DIS structure NODE 1 QUANTITY dispx VALUE  4 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 1 QUANTITY dispy VALUE 4 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 1 QUANTITY dispz VALUE  4 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 22 QUANTITY dispx VALUE  4 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 22 QUANTITY dispy VALUE 4 TOLERANCE 1e-10
        STRUCTURE DIS structure NODE 22 QUANTITY dispz VALUE  4 TOLERANCE 1e-10
        '''
        ))
    
    # Write input file.
    input_file.write_input_file('/home/ivo/temp/curve_lines.dat')



if __name__ == '__main__':
    compare_lines()
    