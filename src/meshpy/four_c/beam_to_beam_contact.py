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
"""This file contains a function to add the beam-to-beam contact conditions for
4C."""

import re
from typing import Optional

import meshpy.core.conf as conf_typing
from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh
from meshpy.four_c.boundary_condition import BoundaryCondition


def get_next_possible_id_for_boundary_condition(
    mesh: Mesh,
    bc_type: conf_typing.BoundaryCondition,
    geometry_type: conf_typing.Geometry,
    regex_search_string: str,
    *,
    group_idx=1,
) -> int:
    """Returns the next possible id, which can be used for a boundary condition
    based on all previous added boundary conditions within a mesh.

    It returns the first ID which is not yet occupied within the existing boundary conditions
    w.r.t. to the given search_string and regex group index.

    Args:
        mesh: Mesh containing already set boundary conditions.
        bc_type: Type of the boundary condition to be searched.
        geometry_type: Geometry type of the boundary condition.
        regex_search_string: String used for the regex search of the bc_string.
        group_idx: Selected index of found regex expression that represents the ID.

    Returns:
        id: Smallest available ID
    """

    found_conditions = []

    # loop through every possible geometry and find the conditions
    for bc_condition in mesh.boundary_conditions[(bc_type, geometry_type)]:
        found_conditions.append(bc_condition)

    if not found_conditions:
        # return starting id, since no conditions of type has been added.
        return 0
    else:
        existing_ids = []

        # compare string of each condition with input and store existing ids
        for bc in found_conditions:
            match = re.search(regex_search_string, bc.bc_string)
            if match is None:
                raise ValueError(
                    "The string provided "
                    + regex_search_string
                    + " could not be found within the condition "
                    + bc.bc_string
                )
            existing_ids.append(int(match.group(group_idx)))

        # return lowest found id
        return min(set(range(len(existing_ids) + 1)) - set(existing_ids))


def add_beam_interaction_condition(
    mesh: Mesh,
    contact_set_1: GeometrySet,
    contact_set_2: GeometrySet,
    id: Optional[int] = None,
) -> int:
    """Adds a pair of beam-to-beam interaction/contact boundary conditions to
    the given mesh and estimates automatically the id of them based on all
    previously added beam- to-beam contact boundary conditions of the mesh.

    Args:
        mesh: Mesh to which the boundary conditions will be added.
        contact_set_1: GeometrySet 1 for contact boundary condition
        contact_set_2: GeometrySet 2 for contact boundary condition
        id: id of the two conditions

    Returns:
        id: used id of the two conditions.
    """

    condition_string = "COUPLING_ID "
    if id is None:
        id = get_next_possible_id_for_boundary_condition(
            mesh,
            mpy.bc.beam_to_beam_contact,
            contact_set_1.geometry_type,
            regex_search_string=re.escape(condition_string) + r"(\d+)",
        )

        id_2 = get_next_possible_id_for_boundary_condition(
            mesh,
            mpy.bc.beam_to_beam_contact,
            contact_set_2.geometry_type,
            regex_search_string=re.escape(condition_string) + r"(\d+)",
        )

        if not id == id_2:
            raise ValueError(
                f"The estimated IDs {id} and {id_2} do not match. Check Inputfile."
            )

    # Creates the two conditions with the same ID.
    for geometry_set in [contact_set_1, contact_set_2]:
        mesh.add(
            BoundaryCondition(
                geometry_set,
                bc_string=condition_string + str(id),
                bc_type=mpy.bc.beam_to_beam_contact,
            )
        )

    return id
