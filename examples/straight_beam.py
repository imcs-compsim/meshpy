# -*- coding: utf-8 -*-
"""
Create a couple of test examples with a straight beam.
"""

# Python modules
import numpy as np

# Meshpy modules.
from meshpy import *


def shear_test():
    """
    """
    
    # Create the beam.
    input_file = InputFile(maintainer='Ivo Steinbrecher')
    input_file.add(
        '''
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
        TIMESTEP                              0.05
        NUMSTEP                               20
        MAXTIME                               1.0
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
        )
    
    # Parameters
    L = 1
    n_el = 1
    diff_y = 0.25
    P = 0.5
    
    # Add material and functions.
    mat = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0.3, 1e-3, 0.2,
        shear_correction=0.75
        )
    ft = Function('COMPONENT 0 FUNCTION t')
    input_file.add(mat, ft)
    
    # Add geometry.
    # First two simply supported beams.
    full_left_1 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0,0,0],
        [0.5*L,0,0],
        n_el=n_el)
    full_right_1 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0.5*L,0,0],
        [L,0,0],
        n_el=n_el, start_node=full_left_1['end'])
    full_left_2 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0,diff_y,0],
        [0.5*L,diff_y,0],
        n_el=n_el)
    full_right_2 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0.5*L,diff_y,0],
        [L,diff_y,0],
        n_el=n_el)
    input_file.add_connections()
    
    # Set boundary conditions.
    for geometry_set in [full_left_1['start'], full_left_2['start'],
            full_right_1['end'], full_right_2['end']]:
        input_file.add(BoundaryCondition(
            geometry_set,
            'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
            bc_type=mpy.dirichlet
            ))
     
    # Set loads.
    input_file.add(BoundaryCondition(
        full_left_1['end'],
        'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 -{} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[P, ft],
        bc_type=mpy.neumann
        ))
     
    # Get middle nodes for second beam.
    def get_node(node):
        """Return true it the nodes is at a certain position."""
        return (np.linalg.norm(node.coordinates - [0.5*L, diff_y, 0])
            < mpy.eps_pos)
     
    node_set = GeometrySet(
        geometry_type = mpy.point,
        nodes = input_file.get_nodes_by_function(get_node)
        )
    input_file.add(BoundaryCondition(
        node_set,
        'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 -{} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[P*0.5, ft],
        bc_type=mpy.neumann
        ))
    
    # Write input file
    input_file.write_input_file('/home/ivo/temp/cantilever.dat')
    
    
    


shear_test()