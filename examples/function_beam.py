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



def curve_2d():
    """ Create a beam from a couple of 2d curves. """
       
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
    
    # Define parametric functions to add to beam.
    def circle(t):
        return np.array([np.cos(t), np.sin(t), t])
    def sin(t):
        return np.array([np.sin(t), t])
    def exp_2(t):
        return np.array([t, np.exp(t)])
    
    # Add the curves to the beam.
    n_el = 50
    sets = []
    sets.append(input_file.create_beam_mesh_curve(Beam3rHerm2Lin3, mat, circle,
        [0,1.9*np.pi], n_el=n_el))
    #input_file.translate([2,0,0])
    #sets.append(input_file.create_beam_mesh_curve_2d(Beam3rHerm2Lin3, mat, sin,
    #    [0,2*np.pi], n_el=n_el))
    #input_file.translate([2,0,0])
    #sets.append(input_file.create_beam_mesh_curve_2d(Beam3rHerm2Lin3, mat, exp_2,
    #    [-1,1], n_el=n_el))
    
    for node_set in sets:
        input_file.add(BoundaryCondition(
            node_set['start'],
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
            bc_type=mpy.dirichlet
            ))
        input_file.add(BoundaryCondition(
            node_set['end'],
            'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
            format_replacement=[0.0001, ft],
            bc_type=mpy.neumann
            ))
    
    # Write input file.
    input_file.write_input_file('/home/ivo/temp/curves.dat')


if __name__ == '__main__':
    curve_2d()
    