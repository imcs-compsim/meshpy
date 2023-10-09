# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2023
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
This module defines functions that can be used to add header information to an
input file.
"""


# Meshpy imports.
from .conf import mpy
from .inputfile import InputSection


def get_yes_no(bool_var):
    """Convert a bool into a string for the baci input file."""
    if bool_var:
        return "yes"
    else:
        return "no"


def get_comment(bool_var):
    """Convert a bool into a comment or no comment for the baci input file."""
    if bool_var:
        return ""
    else:
        return "//"


def _get_segmentation_strategy(segmentation):
    """Get the baci string for a geometry pair strategy."""
    if segmentation:
        return "segmentation"
    else:
        return "gauss_point_projection_without_boundary_segmentation"


def set_runtime_output(
    input_file,
    *,
    output_solid=True,
    output_stress_strain=False,
    btsvmt_output=True,
    btss_output=True,
    output_triad=True,
    every_iteration=False,
    absolute_beam_positons=True,
    element_owner=True,
    element_gid=True,
    output_energy=False,
    output_strains=True,
    option_overwrite=False
):
    """
    Set the basic runtime output options.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    output_solid: bool
        If the solid output should be written at runtime.
    output_stress_strain: bool
        If stress and strain output should be written for the solid.
    btsvmt_output: bool
        If the output for btsvmt should be written.
    btss_output: bool
        If the output for beam-to-surface coupling should be written.
    output_triad: bool
        If the triads along the beam should be written.
    every_iteration: int
        If output at every Newton iteration should be written.
    absolute_beam_positions: bool
        If the beams should be written at the current position or always at
        the reference position.
    element_owner: bool
        If the owing rank of each element should be output (currently
        only affects the solid elements in BACI, beam element owners are
        written by default).
    element_gid: bool
        If the BACI internal GID of each element should be output.
    output_energy: bool
        If the energy output from BACI should be activated.
    output_strains: bool
        If the strains in the Gauss points should be output.
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the basic runtime output options.
    input_file.add(
        InputSection(
            "IO/RUNTIME VTK OUTPUT",
            """
        OUTPUT_DATA_FORMAT        binary
        INTERVAL_STEPS            1
        EVERY_ITERATION           {}""".format(
                get_yes_no(every_iteration)
            ),
            option_overwrite=option_overwrite,
        )
    )

    # Set the structure runtime output options.
    input_file.add(
        InputSection(
            "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            """
        OUTPUT_STRUCTURE                {}
        DISPLACEMENT                    yes
        STRESS_STRAIN                   {}
        ELEMENT_OWNER                   {}
        ELEMENT_GID                     {}""".format(
                get_yes_no(output_solid),
                get_yes_no(output_stress_strain),
                get_yes_no(element_owner),
                get_yes_no(element_gid),
            ),
            option_overwrite=option_overwrite,
        )
    )

    # Set the beam runtime output options.
    input_file.add(
        InputSection(
            "IO/RUNTIME VTK OUTPUT/BEAMS",
            """
        OUTPUT_BEAMS                    yes
        DISPLACEMENT                    yes
        USE_ABSOLUTE_POSITIONS          {}
        TRIAD_VISUALIZATIONPOINT        {}
        STRAINS_GAUSSPOINT              {}
        ELEMENT_GID                     {}""".format(
                get_yes_no(absolute_beam_positons),
                get_yes_no(output_triad),
                get_yes_no(output_strains),
                get_yes_no(element_gid),
            ),
            option_overwrite=option_overwrite,
        )
    )

    if btsvmt_output:
        # Set the beam to solid volume mesh tying runtime output options.
        input_file.add(
            InputSection(
                (
                    "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/"
                    + "RUNTIME VTK OUTPUT"
                ),
                """
            WRITE_OUTPUT                          yes
            NODAL_FORCES                          yes
            MORTAR_LAMBDA_DISCRET                 yes
            MORTAR_LAMBDA_CONTINUOUS              yes
            MORTAR_LAMBDA_CONTINUOUS_SEGMENTS     5
            SEGMENTATION                          yes
            INTEGRATION_POINTS                    yes""",
                option_overwrite=option_overwrite,
            )
        )

    if btss_output:
        # Set the beam to solid surface coupling runtime output options.
        input_file.add(
            InputSection(
                "BEAM INTERACTION/BEAM TO SOLID SURFACE/RUNTIME VTK OUTPUT",
                """
            WRITE_OUTPUT                          yes
            NODAL_FORCES                          yes
            MORTAR_LAMBDA_DISCRET                 yes
            MORTAR_LAMBDA_CONTINUOUS              yes
            MORTAR_LAMBDA_CONTINUOUS_SEGMENTS     5
            SEGMENTATION                          yes
            INTEGRATION_POINTS                    yes
            AVERAGED_NORMALS                      yes""",
                option_overwrite=option_overwrite,
            )
        )

    if output_energy:
        input_file.add("--STRUCTURAL DYNAMIC\nRESEVRYERGY 1")


def set_beam_to_solid_meshtying(
    input_file,
    interaction_type,
    *,
    contact_discretization=None,
    segmentation=True,
    segmentation_search_points=2,
    couple_restart=False,
    mortar_shape=None,
    n_gauss_points=6,
    n_integration_points_circ=None,
    penalty_parameter=None,
    coupling_type=None,
    binning_bounding_box=None,
    binning_cutoff_radius=None,
    option_overwrite=False
):
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
    couple_restart: bool
        If the restart configuration should be used for the coupling
    mortar_shape: str
        Type of shape function for mortar discretization.
    n_gauss_points: int
        Number of Gauss points for numerical integration.
    n_integration_points_circ: int
        Number of integration points along the circumference of the cross
        section.
    penalty_parameter: float
        Penalty parameter for contact enforcement.
    coupling_type: str
        Type of coupling for beam-to-surface coupling.
    binning_bounding_box: [float]
        List with the limits of the bounding box.
    binning_cutoff_radius: float
        Maximal influence radius of pair elements.
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the beam contact options.
    input_file.add(
        InputSection(
            "BEAM INTERACTION", "REPARTITIONSTRATEGY Everydt", option_overwrite=True
        )
    )
    input_file.add(
        InputSection("BEAM CONTACT", "MODELEVALUATOR Standard", option_overwrite=True)
    )

    # Set the binning strategy.
    if (binning_bounding_box is not None) and binning_cutoff_radius is not None:
        bounding_box_string = " ".join([str(val) for val in binning_bounding_box])
        input_file.add(
            InputSection(
                "BINNING STRATEGY",
                """
            BIN_SIZE_LOWER_BOUND {1}
            DOMAINBOUNDINGBOX {0}
            """.format(
                    bounding_box_string, binning_cutoff_radius
                ),
                option_overwrite=True,
            )
        )
    elif (binning_bounding_box is not None) or binning_cutoff_radius is not None:
        raise ValueError(
            (
                "Binning bounding box ({}) and binning cutoff radius"
                + " both have to be set or none of them."
            ).format(binning_bounding_box, binning_cutoff_radius)
        )

    # Add the beam to solid volume mesh tying options.
    if interaction_type == mpy.beam_to_solid.volume_meshtying:
        bts = InputSection("BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING")
    elif interaction_type == mpy.beam_to_solid.surface_meshtying:
        bts = InputSection("BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING")
        if coupling_type is not None:
            bts.add("COUPLING_TYPE {}".format(coupling_type))
    else:
        raise ValueError(
            "Got wrong beam-to-solid mesh tying type. "
            + "Got {} of type {}.".format(interaction_type, type(interaction_type))
        )
    bts.add(
        """
        CONSTRAINT_STRATEGY penalty
        PENALTY_PARAMETER {}
        GAUSS_POINTS {}
        """.format(
            penalty_parameter, n_gauss_points
        ),
        option_overwrite=option_overwrite,
    )
    if contact_discretization == "mortar":
        bts.add(
            """
            CONTACT_DISCRETIZATION mortar
            MORTAR_SHAPE_FUNCTION {}
            """.format(
                mortar_shape
            ),
            option_overwrite=option_overwrite,
        )
        segmentation_strategy = _get_segmentation_strategy(segmentation)
    elif contact_discretization == "gp":
        bts.add(
            "CONTACT_DISCRETIZATION gauss_point_to_segment",
            option_overwrite=option_overwrite,
        )
        segmentation_strategy = _get_segmentation_strategy(segmentation)
    elif contact_discretization == "circ":
        bts.add(
            """
        CONTACT_DISCRETIZATION gauss_point_cross_section
        INTEGRATION_POINTS_CIRCUMFERENCE {}""".format(
                n_integration_points_circ
            ),
            option_overwrite=option_overwrite,
        )
        segmentation_strategy = "gauss_point_projection_cross_section"
    else:
        raise ValueError(
            'Wrong contact_discretization "{}" given!'.format(contact_discretization)
        )

    bts.add(
        """
        GEOMETRY_PAIR_STRATEGY {}
        GEOMETRY_PAIR_SEARCH_POINTS {}
        """.format(
            segmentation_strategy, segmentation_search_points
        ),
        option_overwrite=option_overwrite,
    )
    if couple_restart:
        bts.add("COUPLE_RESTART_STATE yes", option_overwrite=option_overwrite)

    input_file.add(bts)


def set_header_static(
    input_file,
    *,
    time_step=None,
    n_steps=None,
    max_iter=20,
    tol_residuum=1e-8,
    tol_increment=1e-10,
    load_lin=False,
    write_bin=False,
    write_stress="no",
    write_strain="no",
    prestress="none",
    prestress_time=0,
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
    write_bin: bool
        If binary output should be written.
    write_stress: string
        If and which stress output to write
    write_strain: string
        If and which strain output to write
    prestress: string
        Type of prestressing strategy to be used
    presetrss_time: int
        Prestress Time
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the parameters for a static analysis.
    input_file.add(
        InputSection(
            "PROBLEM TYP",
            """
        PROBLEMTYP Structure
        RESTART    0
        """,
            option_overwrite=option_overwrite,
        )
    )
    input_file.add(
        InputSection(
            "IO",
            """
        OUTPUT_BIN     {0}
        STRUCT_DISP    No
        STRUCT_STRESS  {1}
        STRUCT_STRAIN  {2}
        FILESTEPS      1000
        VERBOSITY      Standard
        """.format(
                get_yes_no(write_bin), write_stress, write_strain
            ),
            option_overwrite=option_overwrite,
        )
    )

    input_file.add(
        InputSection(
            "STRUCTURAL DYNAMIC",
            """
        LINEAR_SOLVER     1
        INT_STRATEGY      Standard
        DYNAMICTYP        Statics
        RESULTSEVRY       1
        NLNSOL            fullnewton
        PREDICT           TangDis
        PRESTRESS         {0}
        PRESTRESSTIME     {1}
        TIMESTEP          {2}
        NUMSTEP           {3}
        MAXTIME           {4}
        LOADLIN           {5}
        """.format(
                prestress,
                prestress_time,
                time_step,
                n_steps,
                time_step * n_steps,
                get_yes_no(load_lin),
            ),
            option_overwrite=option_overwrite,
        )
    )
    input_file.add(
        InputSection(
            "SOLVER 1",
            """
        NAME              Structure_Solver
        SOLVER            Superlu
        """,
            option_overwrite=option_overwrite,
        )
    )

    # Set the contents of the NOX xml file.
    nox_xml = """
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
        """.format(
        tol_residuum, tol_increment, max_iter
    )

    input_file.add(
        InputSection(
            "STRUCT NOX/Printing",
            """
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
        """,
            option_overwrite=option_overwrite,
        )
    )

    # Set the xml content in the input file.
    input_file.nox_xml = nox_xml
