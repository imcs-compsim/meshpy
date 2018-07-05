# -*- coding: utf-8 -*-
"""
Create a couple of test examples with a straight beam.
"""

# Python modules
import numpy as np
import autograd.numpy as npAD

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



def cantilever_with_end_load():
    """
    Create a cantilever with an end load in 2D. Three different models are
    created.
    
    1:  The cantilever is one line of beam elements.
    2:  The cantilver is two lines of beam elements with a coupling in between
        of the first 6 DOF.
    3:  The cantilver is two lines of beam elements with a coupling in between
        of the first 9 (all) DOF.
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
    
    # Parameters.
    L = 1
    n_el = 40
    diff_y = 0.0001
    P = 10.
    
    # First the beam with nodal connections.
    sets = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0,0,0],
        [L,0,0],
        n_el=2*n_el)
    
    # Define BCs.
    input_file.add(BoundaryCondition(
        sets['start'],
        'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
        bc_type=mpy.dirichlet
        ))
    input_file.add(BoundaryCondition(
        sets['end'],
        'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[-P, ft],
        bc_type=mpy.neumann
        ))
    
    # Second and third the beams with couplings.
    for i, coupling_type in enumerate([mpy.coupling_fix,
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 1 1 1']):
        sets_left = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
            [0,(1+i)*diff_y,0],
            [0.5*L,(1+i)*diff_y,0],
            n_el=n_el)
        sets_right = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
            [0.5*L,(1+i)*diff_y,0],
            [L,(1+i)*diff_y,0],
            n_el=n_el)
        
        # Define BCs.
        input_file.add(BoundaryCondition(
            sets_left['start'],
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
            bc_type=mpy.dirichlet
            ))
        input_file.add(BoundaryCondition(
            sets_right['end'],
            'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
            format_replacement=[-P, ft],
            bc_type=mpy.neumann
            ))
        
        # Add coupling.
        input_file.couple_nodes(flatten([
                sets_left['line'].nodes,
                sets_right['line'].nodes
            ]),
            coupling_type=coupling_type
            )
    
    # Write input file.
    input_file.write_input_file('/home/ivo/temp/cantilever.dat')



def shear_test():
    """
    Create a beam that is fixed on both sides, with a load in the middle. Five
    different models are created.
    
    1:  The beam is one line of beam elements.
    2:  The beam is two lines of beam elements with a coupling in between of the
        first 6 DOF.
    3:  The beam is two lines of beam elements with a coupling in between of the
        first 9 (all) DOF.
    4:  The beam is only modeled up to the symmetry plane. The bc on the
        symmetry plane is that the shear angle is constrained.
    4:  The beam is only modeled up to the symmetry plane. The bc on the
        symmetry plane is that the shear angle and the beam tangent are
        constrained.
    """
    
    # Create the beam.
    input_file = InputFile(maintainer='Ivo Steinbrecher')
    input_file.add(default_parameters)
    input_file.add('''
        ------------------------STRUCTURAL DYNAMIC
        TIMESTEP                              0.1
        NUMSTEP                               10
        MAXTIME                               1.0
        '''
        )
    
    # Parameters
    L = 1
    n_el = 10
    diff_y = 0.1
    P = 5.
    diameter = 0.25
    
    # Add material and functions.
    mat = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0.3, 1e-3, diameter,
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
    full_left_3 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0,2*diff_y,0],
        [0.5*L,2*diff_y,0],
        n_el=n_el)
    full_right_3 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0.5*L,2*diff_y,0],
        [L,2*diff_y,0],
        n_el=n_el)
    half_1 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0,3*diff_y,0],
        [0.5*L,3*diff_y,0],
        n_el=n_el)
    half_2 = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [0,4*diff_y,0],
        [0.5*L,4*diff_y,0],
        n_el=n_el)
    
    # Couple the middle nodes of the second and third beam.
    input_file.couple_nodes(flatten([
            full_left_2['line'].nodes,
            full_right_2['line'].nodes
        ]),
        coupling_type=mpy.coupling_fix
        )
    input_file.couple_nodes(flatten([
            full_left_3['line'].nodes,
            full_right_3['line'].nodes
        ]),
        coupling_type='NUMDOF 9 ONOFF 1 1 1 1 1 1 1 1 1'
        )
    
    # Set boundary conditions.
    for geometry_set in [full_left_1['start'], full_left_2['start'],
            full_left_3['start'], full_right_1['end'], full_right_2['end'],
            full_right_3['end'], half_1['start'], half_2['start']]:
        input_file.add(BoundaryCondition(
            geometry_set,
            'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
            bc_type=mpy.dirichlet
            ))
     
    # Set load for first beam.
    input_file.add(BoundaryCondition(
        full_left_1['end'],
        'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[-P, ft],
        bc_type=mpy.neumann
        ))
     
    # Get middle nodes for beam.
    def get_node(node, y):
        """Return true it the nodes is at a certain position."""
        return (np.linalg.norm(node.coordinates - [0.5*L, y, 0])
            < mpy.eps_pos)
    
    # Set load for second and third beam.
    node_set = GeometrySet(
        geometry_type = mpy.point,
        nodes = input_file.get_nodes_by_function(get_node, diff_y)
        )
    input_file.add(BoundaryCondition(
        node_set,
        'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[-P, ft],
        bc_type=mpy.neumann, double_nodes=mpy.double_nodes_remove
        ))
    node_set = GeometrySet(
        geometry_type = mpy.point,
        nodes = input_file.get_nodes_by_function(get_node, 2*diff_y)
        )
    input_file.add(BoundaryCondition(
        node_set,
        'NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[-P, ft],
        bc_type=mpy.neumann, double_nodes=mpy.double_nodes_remove
        ))
    
    # Set BC for half beams.
    input_file.add(BoundaryCondition(
        half_1['end'],
        'NUMDOF 9 ONOFF 1 0 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
        bc_type=mpy.dirichlet
        ))
    input_file.add(BoundaryCondition(
        half_1['end'],
        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[-P*0.5, ft],
        bc_type=mpy.neumann
        ))
    input_file.add(BoundaryCondition(
        half_2['end'],
        'NUMDOF 9 ONOFF 1 0 1 1 1 1 1 1 1 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
        bc_type=mpy.dirichlet
        ))
    input_file.add(BoundaryCondition(
        half_2['end'],
        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0',
        format_replacement=[-P*0.5, ft],
        bc_type=mpy.neumann
        ))
        
    # Write input file.
    input_file.write_input_file('/home/ivo/temp/shear-test.dat')


def curve_3d_helix():
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
    n = 2 # number of turns
    n_el = 8
    
    # Sets to apply boundary conditions on.
    sets = []
    
    # Create a line and wrap it around a cylinder.
    sets.append(input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [R, 0, 0], [R, 2*np.pi*n*R, n*tz], n_el=n_el))
    input_file.wrap_around_cylinder()
    
    # Create a helix with a parametric curve.
    offset_x = 2.2 * R
    def helix(t):
        return npAD.array([
            offset_x + R*npAD.cos(t),
            R*npAD.sin(t),
            t*tz/(2*np.pi)
            ])
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
    input_file.write_input_file('/home/ivo/temp/curve_3d_helix.dat')


def curve_3d_line_rotation():
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
    offset = 0.5*L
    def line(t):
        return npAD.array([L - t*L, offset, 0.])
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
            format_replacement=[0.25, ft, 0.025],
            bc_type=mpy.neumann
            ))
    
    # Write input file.
    input_file.write_input_file('/home/ivo/temp/curve_3d_line_rotation.dat')


if __name__ == '__main__':
    shear_test()
    cantilever_with_end_load()
    curve_3d_helix()
    curve_3d_line_rotation()