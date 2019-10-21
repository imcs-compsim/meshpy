# -*- coding: utf-8 -*-
"""
This module defines functions that can be used to add header information to an
input file.
"""


# Meshpy imports.
from .conf import mpy
from .inputfile import InputSection


def _get_yes_no(bool_var):
    """Convert a bool into a string for the baci input file."""
    if bool_var:
        return 'yes'
    else:
        return 'no'


def _get_segmentation_strategy(segmentation):
    """Get the baci string for a geometry pair strategy."""
    if segmentation:
        return 'segmentation'
    else:
        return 'gauss_point_projection'


def set_runtime_output(input_file, *,
        output_solid=True,
        btsvmt_output=True,
        output_triad=True,
        every_iteration=False,
        option_overwrite=False):
    """
    Set the basic runtime output options.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    output_solid: bool
        If the solid output should be written at runtime.
    btsvmt_output: bool
        If the output for btsvmt should be written.
    output_triad: bool
        If the triads along the beam should be written.
    every_iteration: int
        If output at every Newton iteration should be written.
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the basic runtime output options.
    input_file.add(InputSection(
        'IO/RUNTIME VTK OUTPUT',
        '''
        OUTPUT_DATA_FORMAT        binary
        INTERVAL_STEPS            1
        EVERY_ITERATION           {}'''.format(_get_yes_no(every_iteration)),
        option_overwrite=option_overwrite))

    # Set the structure runtime output options.
    input_file.add(InputSection(
        'IO/RUNTIME VTK OUTPUT/STRUCTURE',
        '''
        OUTPUT_STRUCTURE                {}
        DISPLACEMENT                    yes
        ELEMENT_OWNER                   yes'''.format(
            _get_yes_no(output_solid)),
        option_overwrite=option_overwrite))

    # Set the beam runtime output options.
    input_file.add(InputSection(
        'IO/RUNTIME VTK OUTPUT/BEAMS',
        '''
        OUTPUT_BEAMS                    yes
        DISPLACEMENT                    yes
        USE_ABSOLUTE_POSITIONS          yes
        TRIAD_VISUALIZATIONPOINT        {}
        STRAINS_GAUSSPOINT              yes'''.format(
            _get_yes_no(output_triad)),
        option_overwrite=option_overwrite))

    if btsvmt_output:
        # Set the beam to solid volume mesh tying runtime output options.
        input_file.add(InputSection(
            ('BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/'
                + 'RUNTIME VTK OUTPUT'),
            '''
            WRITE_OUTPUT                          yes
            NODAL_FORCES                          yes
            MORTAR_LAMBDA_DISCRET                 yes
            MORTAR_LAMBDA_CONTINUOUS              yes
            MORTAR_LAMBDA_CONTINUOUS_SEGMENTS     5
            SEGMENTATION                          yes
            INTEGRATION_POINTS                    yes''',
            option_overwrite=option_overwrite))


def set_beam_to_solid_meshtying(input_file, interaction_type, *,
        contact_discretization=None,
        segmentation=True,
        segmentation_search_points=2,
        mortar_shape=None,
        n_gauss_points=6,
        n_integration_points_circ=None,
        penalty_parameter=None,
        binning_bounding_box=None,
        binning_cutoff_radius=-1,
        option_overwrite=False):
    """
    Set the beam to solid meshtying options.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    interaction_type: BeamToSolidInteractionType
        Type of beam-to-solid interation.
    contact_discretization: str
        Type of contact (mortar, Gauss point, ...)
    segmentation: bool
        If segmentation should be used in the numerical integration.
    segmentation_search_points: int
        Number of search points for segmentation.
    mortar_shape: str
        Type of shape function for mortar discretization.
    n_gauss_points: int
        Number of Gauss points for numerical integration.
    n_integration_points_circ: int
        Number of integration points along the circumference of the cross
        section.
    penalty_parameter: float
        Penalty parameter for contact enforcement.
    binning_bounding_box: [float]
        List with the limits of the bounding box.
    binning_cutoff_radius: float
        Maximal influence radius of pair elements.
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the beam contact options.
    input_file.add(InputSection(
        'BEAM INTERACTION', 'REPARTITIONSTRATEGY Everydt',
        option_overwrite=True))
    input_file.add(InputSection(
        'BEAM CONTACT', 'MODELEVALUATOR Standard',
        option_overwrite=True))

    # Set the binning strategy.
    bounding_box_string = ' '.join([str(val) for val in binning_bounding_box])
    input_file.add(InputSection(
        'BINNING STRATEGY',
        '''
        CUTOFF_RADIUS {1}
        BOUNDINGBOX {0}
        '''.format(bounding_box_string, binning_cutoff_radius),
        option_overwrite=True))

    # Add the beam to solid volume mesh tying options.
    if interaction_type == mpy.beam_to_solid.volume_meshtying:
        bts = InputSection('BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING')
    elif interaction_type == mpy.beam_to_solid.surface_meshtying:
        bts = InputSection('BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING')
    bts.add('''
        CONSTRAINT_STRATEGY penalty
        PENALTY_PARAMETER {}
        GAUSS_POINTS {}
        '''.format(penalty_parameter, n_gauss_points),
        option_overwrite=option_overwrite)
    if contact_discretization is 'mortar':
        bts.add('''
            CONTACT_DISCRETIZATION mortar
            MORTAR_SHAPE_FUNCTION {}
            '''.format(mortar_shape),
            option_overwrite=option_overwrite)
        segmentation_strategy = _get_segmentation_strategy(segmentation)
    elif contact_discretization is 'gp':
        bts.add('CONTACT_DISCRETIZATION gauss_point_to_segment',
            option_overwrite=option_overwrite)
        segmentation_strategy = _get_segmentation_strategy(segmentation)
    elif contact_discretization is 'circ':
        bts.add('''
        CONTACT_DISCRETIZATION gauss_point_cross_section
        INTEGRATION_POINTS_CIRCUMFENCE {}'''.format(n_integration_points_circ),
            option_overwrite=option_overwrite)
        segmentation_strategy = 'gauss_point_projection_cross_section'
    else:
        raise ValueError('Wrong contact_discretization "{}" given!'.format(
            contact_discretization))
    bts.add(
        '''
        GEOMETRY_PAIR_STRATEGY {}
        GEOMETRY_PAIR_SEARCH_POINTS {}
        '''.format(segmentation_strategy, segmentation_search_points))
    input_file.add(bts)


def set_header_static(input_file, *,
        time_step=None,
        n_steps=None,
        max_iter=20,
        tol_residuum=1e-8,
        tol_increment=1e-10,
        load_lin=False,
        option_overwrite=False
        ):
    """
    Set the default parameters for a static structure analysis.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    time_step: float
        Time increment per step.
    n_steps: int
        Number of time steps.
    max_iter: int
        Maximal number of Newton iterations.
    tol_residuum: float
        Tolerance for the convergence of the residuum.
    tol_increment: int
        Tolerance for the convergence of the displacement increment.
    load_lin: bool
        If the load_lin option should be set.
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the parameters for a static analysis.
    input_file.add(InputSection('PROBLEM TYP',
        '''
        PROBLEMTYP Structure
        RESTART    0
        ''',
        option_overwrite=option_overwrite))
    input_file.add(InputSection('IO',
        '''
        OUTPUT_BIN     No
        STRUCT_DISP    No
        FILESTEPS      1000
        VERBOSITY      Standard
        ''',
        option_overwrite=option_overwrite))

    input_file.add(InputSection(
        'STRUCTURAL DYNAMIC',
        '''
        LINEAR_SOLVER     1
        INT_STRATEGY      Standard
        DYNAMICTYP        Statics
        RESULTSEVRY       1
        NLNSOL            fullnewton
        PREDICT           TangDis
        TIMESTEP          {0}
        NUMSTEP           {1}
        MAXTIME           {2}
        LOADLIN           {3}
        '''.format(time_step, n_steps, time_step * n_steps,
            _get_yes_no(load_lin)),
        option_overwrite=option_overwrite))
    input_file.add(InputSection(
        'SOLVER 1',
        '''
        NAME              Structure_Solver
        SOLVER            Superlu
        ''',
        option_overwrite=option_overwrite))

    # Set the contents of the NOX xml file.
    nox_xml = '''
        <ParameterList name="Status Test">
        <!-- Outer Status Test: This test is an OR combination of the structural convergence and the maximum number of iterations -->
        <ParameterList name="Outer Status Test">
          <Parameter name="Test Type"       type="string" value="Combo"/>
          <Parameter name="Combo Type"      type="string" value="OR" />
          <!-- Structural convergence is an AND combination of the residuum and step update -->
          <ParameterList name="Test 0">
            <Parameter name="Test Type" type="string" value="Combo" />
            <Parameter name="Combo Type" type="string" value="AND" />
              <!-- BEGIN: Combo AND - Test 0: "NormF" -->
              <ParameterList name="Test 0">
                <Parameter name="Test Type"  type="string" value="NormF" />
                <!-- NormF - Quantity 0: Check the right-hand-side norm of the structural quantities -->
                <ParameterList name="Quantity 0">
                  <Parameter name="Quantity Type"  type="string" value="Structure" />
                  <Parameter name="Tolerance Type" type="string" value="Absolute" />
                  <Parameter name="Tolerance"      type="double" value="{0}" />
                  <Parameter name="Norm Type"      type="string" value="Two Norm" />
                  <Parameter name="Scale Type"     type="string" value="Scaled" />
                </ParameterList>
              </ParameterList>
              <!-- END: Combo AND - Test 0: "NormF" -->
              <!-- BEGIN: Combo AND - Test 1: "NormWRMS" -->
              <ParameterList name="Test 1">
                <Parameter name="Test Type"        type="string" value="NormUpdate" />
                <!-- NormWRMS - Quantity 0: Check the increment of the structural displacements -->
                <ParameterList name="Quantity 0">
                  <Parameter name="Quantity Type"  type="string" value="Structure" />
                  <Parameter name="Tolerance Type" type="string" value="Absolute" />
                  <Parameter name="Tolerance"      type="double" value="{1}" />
                  <Parameter name="Norm Type"      type="string" value="Two Norm" />
                  <Parameter name="Scale Type"     type="string" value="Scaled" />
                </ParameterList>
              </ParameterList>
              <!-- END: Combo AND - Test 1: "NormWRMS" -->
            </ParameterList>
            <!-- END: Combo 0 - Test 0: "Combo" -->
          <!-- BEGIN: Combo OR - Test 1: "MaxIters" -->
          <ParameterList name="Test 1">
            <Parameter name="Test Type"          type="string" value="MaxIters" />
            <Parameter name="Maximum Iterations" type="int"    value="{2}" />
          </ParameterList> <!--END: "MaxIters" -->
        </ParameterList>
        </ParameterList>
        '''.format(tol_residuum, tol_increment, max_iter)

    input_file.add(InputSection('STRUCT NOX/Printing',
        '''
        Error                           = Yes
        Warning                         = Yes
        Outer Iteration                 = Yes
        Inner Iteration                 = No
        Parameters                      = No
        Details                         = Yes
        Outer Iteration StatusTest      = Yes
        Linear Solver Details           = Yes
        Test Details                    = Yes
        Debug                           = No
        ''',
        option_overwrite=option_overwrite))

    # Set the xml content in the input file.
    input_file.nox_xml = nox_xml
