# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
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
"""This file includes functions to ease the creation of input files using beam
interaction potentials."""


# Python modules
import numpy as np

# MeshPy modules
from ..boundary_condition import BoundaryCondition
from ..header_functions import get_yes_no
from ..inputfile import InputSection


class BeamPotential:
    """Class which provides functions for the usage of beam to beam potential
    interactions within 4C based on a potential law in form of a power law."""

    def __init__(
        self,
        input_file,
        *,
        pot_law_prefactor=None,
        pot_law_exponent=None,
        pot_law_line_charge_density=None,
        pot_law_line_charge_density_funcs=None,
    ):
        """Initialize object to enable beam potential interactions.

        Args
        ----
        input_file:
            Input file of current problem setup.
        pot_law_prefactors: float, int, np.array, list
            Prefactors of a potential law in form of a power law. Same number
            of prefactors and exponents/line charge densities/functions must be
            provided!
        pot_law_exponent: float, int, np.array, list
            Exponents of a potential law in form of a power law. Same number
            of exponents and prefactors/line charge densities/functions must be
            provided!
        pot_law_line_charge_density: float, int, np.array, list
            Line charge densities of a potential law in form of a power law.
            Same number of line charge densities and prefactors/exponents/functions
            must be provided!
        pot_law_line_charge_density_funcs:
            Functions for line charge densities of a potential law in form of a
            power law. Same number of functions and prefactors/exponents/line
            charge densities must be provided!
        """

        self.input_file = input_file

        # if only one potential law prefactor/exponent is present, convert it
        # into a list for simplified usage
        if isinstance(pot_law_prefactor, (float, int)):
            pot_law_prefactor = [pot_law_prefactor]
        if isinstance(pot_law_exponent, (float, int)):
            pot_law_exponent = [pot_law_exponent]
        if isinstance(pot_law_line_charge_density, (float, int)):
            pot_law_line_charge_density = [pot_law_line_charge_density]

        # check if same number of prefactors and exponents are provided
        if (
            not len(pot_law_prefactor)
            == len(pot_law_exponent)
            == len(pot_law_line_charge_density)
        ):
            raise ValueError(
                "Number of potential law prefactors do not match potential law exponents!"
            )

        self.pot_law_prefactor = pot_law_prefactor
        self.pot_law_exponent = pot_law_exponent
        self.pot_law_line_charge_density = pot_law_line_charge_density
        self.pot_law_line_charge_density_funcs = pot_law_line_charge_density_funcs

    def add_header(
        self,
        *,
        potential_type="Volume",
        cutoff_radius=None,
        evaluation_strategy=None,
        regularization_type=None,
        regularization_separation=None,
        integration_segments=1,
        gauss_points=10,
        potential_reduction_length=-1,
        automatic_differentiation=False,
        choice_master_slave=None,
        option_overwrite=False,
    ):
        """Set the basic header options for beam potential interactions.

        Args
        ----
        potential_type: string
            Type of applied potential (volume, surface).
        cutoff_radius: float
            Neglect all contributions at separation larger than this cutoff
            radius.
        evaluation_strategy: string
            Strategy to evaluate interaction potential.
        regularization_type: string
            Type of regularization to use for force law at separations below
            specified separation (constant_extrapolation, linear_extrapolation).
        regularization_separation: float
            Use specified regularization type for separations smaller than
            this value.
        integration_segments: int
            Number of integration segments to be used per beam element.
        gauss_points: int
            Number of Gauss points to be used per integration segment.
        potential_reduction_length: float
            Potential is smoothly decreased within this length when using the
            single length specific (SBIP) approach to enable an axial pull off
            force.
        automatic_differentiation: bool
            Use automatic differentiation via FAD.
        choice_master_slave: string
            Rule how to assign the role of master and slave to beam elements (if
            applicable) (lower_eleGID_is_slave, higher_eleGID_is_slave).
        option_overwrite: bool
            If existing options should be overwritten. If this is false and an
            option is already defined, and error will be thrown.
        """

        settings = f"""
            POT_LAW_PREFACTOR                     {' '.join(map(str, self.pot_law_prefactor))}
            POT_LAW_EXPONENT                      {' '.join(map(str, self.pot_law_exponent))}
            BEAMPOTENTIAL_TYPE              {potential_type}
            CUTOFF_RADIUS                   {cutoff_radius}
            STRATEGY                        {evaluation_strategy}
            NUM_INTEGRATION_SEGMENTS        {integration_segments}
            NUM_GAUSSPOINTS                 {gauss_points}
            POTENTIAL_REDUCTION_LENGTH      {potential_reduction_length}
            AUTOMATIC_DIFFERENTIATION       {get_yes_no(automatic_differentiation)}"""

        if regularization_type is not None:
            settings += f"""
            REGULARIZATION_TYPE             {regularization_type}
            REGULARIZATION_SEPARATION       {regularization_separation}"""

        if choice_master_slave is not None:
            settings += f"\nCHOICE_MASTER_SLAVE             {choice_master_slave}"

        self.input_file.add(
            InputSection(
                "BEAM POTENTIAL",
                settings,
                option_overwrite=option_overwrite,
            )
        )

    def add_runtime_output(
        self,
        *,
        output_beam_potential=True,
        interval_steps=1,
        every_iteration=False,
        forces=True,
        moments=True,
        per_ele_pair=True,
        option_overwrite=False,
    ):
        """Set the basic runtime output options for beam potential
        interactions.

        Args
        ----
        output_beam_potential: bool
            If the output for beam potential should be written.
        interval_steps: int
            Interval at which output is written.
        every_iteration: bool
            If output at every Newton iteration should be written.
        forces: bool
            If the forces should be written.
        moments: bool
            If the moments should be written.
        per_ele_pair: bool
            If the forces/moments should be written per element pair.
        option_overwrite: bool
            If existing options should be overwritten. If this is false and an
            option is already defined, and error will be thrown.
        """

        self.input_file.add(
            InputSection(
                "BEAM POTENTIAL/RUNTIME VTK OUTPUT",
                f"""
            VTK_OUTPUT_BEAM_POTENTIAL           {get_yes_no(output_beam_potential)}
            INTERVAL_STEPS                      {interval_steps}
            EVERY_ITERATION                     {get_yes_no(every_iteration)}
            FORCES                              {get_yes_no(forces)}
            MOMENTS                             {get_yes_no(moments)}
            WRITE_FORCE_MOMENT_PER_ELEMENTPAIR  {get_yes_no(per_ele_pair)}""",
                option_overwrite=option_overwrite,
            )
        )

    def add_potential_charge_condition(self, *, geometry_set=None):
        """Add potential charge condition to geometry.

        Args
        ----
        geometry_set:
            Add potential charge condition to this set.
        """

        for i, (line_charge, func) in enumerate(
            zip(
                self.pot_law_line_charge_density, self.pot_law_line_charge_density_funcs
            )
        ):

            if func != "none":
                self.input_file.add(func)

            bc = BoundaryCondition(
                geometry_set,
                f"POTLAW {i+1} VAL {line_charge} FUNCT {{}}",
                bc_type="DESIGN LINE BEAM POTENTIAL CHARGE CONDITIONS",
                format_replacement=[func],
            )

            self.input_file.add(bc)
