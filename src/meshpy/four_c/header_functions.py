# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This module defines functions that can be used to add header information to
an input file."""

from typing import List, Optional, Union

from meshpy.core.conf import mpy as _mpy
from meshpy.four_c.input_file import InputFile as _InputFile
from meshpy.four_c.input_file import InputSection as _InputSection


def get_yes_no(bool_var):
    """Convert a bool into a string for the 4C input file."""
    if bool_var:
        return "yes"
    else:
        return "no"


def get_comment(bool_var):
    """Convert a bool into a comment or no comment for the 4C input file."""
    if bool_var:
        return ""
    else:
        return "//"


def _get_segmentation_strategy(segmentation):
    """Get the 4C string for a geometry pair strategy."""
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
    absolute_beam_positions=True,
    element_owner=True,
    element_gid=True,
    element_mat_id=True,
    output_energy=False,
    output_strains=True,
    option_overwrite=False,
):
    """Set the basic runtime output options.

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
        only affects the solid elements in 4C, beam element owners are
        written by default).
    element_gid: bool
        If the 4C internal GID of each element should be output.
    element_mat_id: bool
        If the 4C internal material ID of each element should be output.
    output_energy: bool
        If the energy output from 4C should be activated.
    output_strains: bool
        If the strains in the Gauss points should be output.
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the basic runtime output options.
    input_file.add(
        _InputSection(
            "IO/RUNTIME VTK OUTPUT",
            f"""
        OUTPUT_DATA_FORMAT        binary
        INTERVAL_STEPS            1
        EVERY_ITERATION           {get_yes_no(every_iteration)}""",
            option_overwrite=option_overwrite,
        )
    )

    # Set the structure runtime output options.
    input_file.add(
        _InputSection(
            "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            f"""
        OUTPUT_STRUCTURE                {get_yes_no(output_solid)}
        DISPLACEMENT                    yes
        STRESS_STRAIN                   {get_yes_no(output_stress_strain)}
        ELEMENT_OWNER                   {get_yes_no(element_owner)}
        ELEMENT_GID                     {get_yes_no(element_gid)}
        ELEMENT_MAT_ID                  {get_yes_no(element_mat_id)}""",
            option_overwrite=option_overwrite,
        )
    )

    # Set the beam runtime output options.
    input_file.add(
        _InputSection(
            "IO/RUNTIME VTK OUTPUT/BEAMS",
            f"""
        OUTPUT_BEAMS                    yes
        DISPLACEMENT                    yes
        USE_ABSOLUTE_POSITIONS          {get_yes_no(absolute_beam_positions)}
        TRIAD_VISUALIZATIONPOINT        {get_yes_no(output_triad)}
        STRAINS_GAUSSPOINT              {get_yes_no(output_strains)}
        ELEMENT_GID                     {get_yes_no(element_gid)}""",
            option_overwrite=option_overwrite,
        )
    )

    if btsvmt_output:
        # Set the beam to solid volume mesh tying runtime output options.
        input_file.add(
            _InputSection(
                ("BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/RUNTIME VTK OUTPUT"),
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
            _InputSection(
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
        input_file.add("--STRUCTURAL DYNAMIC\nRESEVERYERGY 1")


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
    binning_parameters: dict = {},
    option_overwrite=False,
):
    """Set the beam to solid meshtying options.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    interaction_type: BeamToSolidInteractionType
        Type of beam-to-solid interaction.
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
    binning_parameters:
        Keyword parameters for the binning section
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    # Set the beam contact options.
    input_file.add(
        _InputSection(
            "BEAM INTERACTION", "REPARTITIONSTRATEGY Everydt", option_overwrite=True
        )
    )
    input_file.add(
        _InputSection("BEAM CONTACT", "MODELEVALUATOR Standard", option_overwrite=True)
    )

    set_binning_strategy_section(
        input_file,
        option_overwrite=option_overwrite,
        **binning_parameters,
    )

    # Add the beam to solid volume mesh tying options.
    if interaction_type == _mpy.beam_to_solid.volume_meshtying:
        bts = _InputSection("BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING")
    elif interaction_type == _mpy.beam_to_solid.surface_meshtying:
        bts = _InputSection("BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING")
        if coupling_type is not None:
            bts.add(f"COUPLING_TYPE {coupling_type}")
    else:
        raise ValueError(
            "Got wrong beam-to-solid mesh tying type. "
            f"Got {interaction_type} of type {type(interaction_type)}."
        )
    bts.add(
        f"""
        CONSTRAINT_STRATEGY penalty
        PENALTY_PARAMETER {penalty_parameter}
        GAUSS_POINTS {n_gauss_points}
        """,
        option_overwrite=option_overwrite,
    )
    if contact_discretization == "mortar":
        bts.add(
            f"""
            CONTACT_DISCRETIZATION mortar
            MORTAR_SHAPE_FUNCTION {mortar_shape}
            """,
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
            f"""
        CONTACT_DISCRETIZATION gauss_point_cross_section
        INTEGRATION_POINTS_CIRCUMFERENCE {n_integration_points_circ}""",
            option_overwrite=option_overwrite,
        )
        segmentation_strategy = "gauss_point_projection_cross_section"
    else:
        raise ValueError(
            f'Wrong contact_discretization "{contact_discretization}" given!'
        )

    bts.add(
        f"""
        GEOMETRY_PAIR_STRATEGY {segmentation_strategy}
        GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS {segmentation_search_points}
        """,
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
    total_time=None,
    max_iter=20,
    tol_residuum=1e-8,
    tol_increment=1e-10,
    load_lin=False,
    write_bin=False,
    write_stress="no",
    write_strain="no",
    prestress="none",
    prestress_time=0,
    option_overwrite=False,
):
    """Set the default parameters for a static structure analysis.

    At least two of the three time stepping keyword arguments ["time_step",
    "n_steps", "total_time"] have to be set.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    time_step: float
        Time increment per step.
    n_steps: int
        Number of time steps.
    total_time: float
        Total time of simulation
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
        _InputSection(
            "PROBLEM TYPE",
            """
        PROBLEMTYPE Structure
        RESTART    0
        """,
            option_overwrite=option_overwrite,
        )
    )
    input_file.add(
        _InputSection(
            "IO",
            f"""
        OUTPUT_BIN     {get_yes_no(write_bin)}
        STRUCT_DISP    No
        STRUCT_STRESS  {write_stress}
        STRUCT_STRAIN  {write_strain}
        FILESTEPS      1000
        VERBOSITY      Standard
        """,
            option_overwrite=option_overwrite,
        )
    )

    # Set the time step parameters
    given_time_arguments = sum(
        1 for arg in (time_step, n_steps, total_time) if arg is not None
    )
    if given_time_arguments < 2:
        raise ValueError(
            'At least two of the following arguments "time_step", "n_steps" or '
            '"total_time" are required'
        )
    if time_step is None:
        time_step = total_time / n_steps
    elif n_steps is None:
        n_steps = round(total_time / time_step)
    elif total_time is None:
        total_time = time_step * n_steps

    input_file.add(
        _InputSection(
            "STRUCTURAL DYNAMIC",
            f"""
        LINEAR_SOLVER     1
        INT_STRATEGY      Standard
        DYNAMICTYPE       Statics
        RESULTSEVERY      1
        NLNSOL            fullnewton
        PREDICT           TangDis
        PRESTRESS         {prestress}
        PRESTRESSTIME     {prestress_time}
        TIMESTEP          {time_step}
        NUMSTEP           {n_steps}
        MAXTIME           {total_time}
        LOADLIN           {get_yes_no(load_lin)}
        """,
            option_overwrite=option_overwrite,
        )
    )
    input_file.add(
        _InputSection(
            "SOLVER 1",
            """
        NAME              Structure_Solver
        SOLVER            Superlu
        """,
            option_overwrite=option_overwrite,
        )
    )

    # Set the contents of the NOX xml file.
    nox_xml = f"""
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
                  <Parameter name="Tolerance"      type="double" value="{tol_residuum}" />
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
                  <Parameter name="Tolerance"      type="double" value="{tol_increment}" />
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
            <Parameter name="Maximum Iterations" type="int"    value="{max_iter}" />
          </ParameterList> <!--END: "MaxIters" -->
        </ParameterList>
        </ParameterList>
        """

    input_file.add(
        _InputSection(
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


def set_binning_strategy_section(
    input_file: _InputFile,
    binning_bounding_box: Union[List[int], None] = None,
    binning_cutoff_radius: Union[float, None] = None,
    *,
    option_overwrite: bool = False,
):
    """Set binning strategy in section of the input file.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    binning_bounding_box:
        List with the limits of the bounding box.
    binning_cutoff_radius:
        Maximal influence radius of pair elements.
    option_overwrite:
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    if binning_bounding_box is not None and binning_cutoff_radius is not None:
        binning_bounding_box_string = " ".join(
            [str(val) for val in binning_bounding_box]
        )

        input_file.add(
            _InputSection(
                "BINNING STRATEGY",
                f"""
            BIN_SIZE_LOWER_BOUND    {binning_cutoff_radius}
            DOMAINBOUNDINGBOX                     {binning_bounding_box_string}
            """,
                option_overwrite=option_overwrite,
            )
        )
    elif [binning_bounding_box, binning_cutoff_radius].count(None) == 2:
        return
    else:
        raise ValueError(
            f"The variables binning_bounding_box {binning_bounding_box} and binning_cutoff_radius {binning_cutoff_radius} must both be set."
        )


def set_beam_interaction_section(
    inputfile: _InputFile,
    *,
    repartition_strategy: str = "everydt",
    search_strategy: str = "bounding_volume_hierarchy",
    option_overwrite: bool = False,
):
    """Set beam interaction section in input file.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    repartition_strategy:
        Type of employed repartitioning strategy
        Options: "adaptive" or "everydt"
    search_strategy:
        Type of search strategy used for finding coupling pairs.
        Options: "bruteforce_with_binning", "bounding_volume_hierarchy"
    option_overwrite:
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    inputfile.add(
        _InputSection(
            "BEAM INTERACTION",
            f"""
        REPARTITIONSTRATEGY                   {repartition_strategy}
        SEARCH_STRATEGY                       {search_strategy}
        """,
            option_overwrite=option_overwrite,
        )
    )


def set_beam_contact_runtime_output(
    inputfile: _InputFile,
    *,
    every_iteration: bool = False,
    option_overwrite: bool = False,
):
    """Output the beam-to-beam contact forces and gaps with runtime output.

    input_file:
        Input file that the options will be added to.
    every_iteration:
        If output at every Newton iteration should be written.
    option_overwrite:
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    """

    inputfile.add(
        _InputSection(
            "BEAM CONTACT/RUNTIME VTK OUTPUT",
            f"""
            VTK_OUTPUT_BEAM_CONTACT               yes
            EVERY_ITERATION                       {get_yes_no(every_iteration)}
            INTERVAL_STEPS                        1
            CONTACT_FORCES                        yes
            GAPS                                  yes
        """,
            option_overwrite=option_overwrite,
        )
    )


def set_beam_contact_section(
    input_file: _InputFile,
    *,
    interaction_strategy: str = "penalty",
    btb_penalty: float = 0,
    btb_line_penalty: float = 0,
    per_shift_angle: list[float] = [70, 80],
    par_shift_angle: list[float] = [70, 80],
    b_seg_angle: float = 12,
    num_integration: int = 5,
    penalty_law: str = "LinPosQuadPen",
    penalty_regularization_g0: float = 0,
    penalty_regularization_f0: float = 0,
    penalty_regularization_c0: float = 0,
    binning_parameters: dict = {},
    beam_interaction_parameters: dict = {},
    option_overwrite: bool = False,
):
    """Set default beam contact section, for more and updated details see
    respective input file within 4C. Parameters for set_binning_strategy and
    set_beam_interaction may be forwarded as keyword arguments.

    Args
    ----
    input_file:
        Input file that the options will be added to.
    interaction_strategy:
        Type of employed solving strategy
        Options: "none", "penalty" or "gmshonly"
    btb_penalty: double
        Penalty parameter for beam-to-beam point contact
    btb_line_penalty:
        Penalty parameter per unit length for beam-to-beam line contact
    per_shift_angle:
        Lower and upper shift angle (in degrees) for penalty scaling of large-angle-contact
    par_shift_angle:
        Lower and upper shift angle (in degrees) for penalty scaling of small-angle-contact
    b_seg_angle:
        Maximal angle deviation allowed for contact search segmentation
    num_integration:
        Number of integration intervals per element
    option_overwrite: bool
        If existing options should be overwritten. If this is false and an
        option is already defined, and error will be thrown.
    penalty_law:
        Penalty Law Options: "LinPen", "QuadPen", "LinNegQuadPen", "LinPosQuadPen", "LinPosCubPen", "LinPosDoubleQuadPen", "LinPosExpPen"
    penalty_regularization_g0:
        First penalty regularization parameter G0
    penalty_regularization_f0:
        Second penalty regularization parameter F0
    penalty_regularization_c0:
        Third penalty regularization parameter C0
    binning_parameters:
        Keyword parameters for the binning section
    beam_interaction_parameters:
        Keyword parameters for the beam-contact section
    """

    if len(per_shift_angle) != 2:
        raise ValueError(
            "Please provide lower and upper value of BEAMS_PERPSHIFTANGLE."
        )

    if len(par_shift_angle) != 2:
        raise ValueError("Please provide lower and upper value of BEAMS_PARSHIFTANGLE.")

    input_file.add(
        _InputSection(
            "BEAM INTERACTION/BEAM TO BEAM CONTACT",
            f"""STRATEGY   {interaction_strategy}""",
            option_overwrite=option_overwrite,
        )
    )
    input_file.add(
        _InputSection(
            "BEAM CONTACT",
            f"""MODELEVALUATOR                  Standard
        BEAMS_STRATEGY                  Penalty
        BEAMS_BTBPENALTYPARAM           {btb_penalty}
        BEAMS_BTBLINEPENALTYPARAM       {btb_line_penalty}
        BEAMS_SEGCON                    Yes
        BEAMS_PERPSHIFTANGLE1           {per_shift_angle[0]}
        BEAMS_PERPSHIFTANGLE2           {per_shift_angle[1]}
        BEAMS_PARSHIFTANGLE1            {par_shift_angle[0]}
        BEAMS_PARSHIFTANGLE2            {par_shift_angle[1]}
        BEAMS_SEGANGLE                  {b_seg_angle}
        BEAMS_NUMINTEGRATIONINTERVAL    {num_integration}
        BEAMS_PENALTYLAW                {penalty_law}
        BEAMS_PENREGPARAM_G0            {penalty_regularization_g0}
        BEAMS_PENREGPARAM_F0            {penalty_regularization_f0}
        BEAMS_PENREGPARAM_C0            {penalty_regularization_c0}
        BEAMS_MAXDISISCALEFAC           -1.0
        BEAMS_MAXDELTADISSCALEFAC       -1.0
        """,
            option_overwrite=option_overwrite,
        )
    )

    # beam contact needs a binning strategy
    set_binning_strategy_section(
        input_file, option_overwrite=option_overwrite, **binning_parameters
    )

    # beam contact needs interaction strategy
    set_beam_interaction_section(
        input_file, option_overwrite=option_overwrite, **beam_interaction_parameters
    )
