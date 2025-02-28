# -*- coding: utf-8 -*-
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This file contains a function to add the beam-to-beam contact conditions for
4C."""

import re

from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh
from meshpy.four_c.boundary_condition import BoundaryCondition


def get_next_possible_id_for_boundary_condition(
    mesh: Mesh, bc_type: BoundaryCondition, search_string, group_idx=1
):
    """Returns the next possible id, which can be used for a boundary condition
    based on all previous added boundary conditions within a mesh. It estimates
    the id based on the bc_string wrt to the given search_string and regex
    group index.

    Args:
        mesh (Mesh): Mesh containing already set boundary conditions.
        bc_type (mpy.bc): Type of the boundary condition to be searched.
        search_string (string): string searched of the bc_string
        group_idx(int): selected argument of regex expression

    Returns:
        id (int): smallest used id
    """

    found_conditions = []

    # loop through every possible geometry and find the conditions
    for geometry_type in mpy.geo:
        for bc_condition in mesh.boundary_conditions[(bc_type, geometry_type)]:
            found_conditions.append(bc_condition)

    if not found_conditions:
        # return starting id, since no conditions of type has been added.
        return 0
    else:
        existing_ids = []

        # compare string of each condition with input and store existing ids
        for bc in found_conditions:
            match = re.search(search_string, bc.bc_string)
            assert match is not None
            existing_ids.append(int(match.group(group_idx)))

        # return lowest found id
        return min(set(range(len(existing_ids) + 1)) - set(existing_ids))


def add_contact_boundary_condition_to_mesh(
    mesh: Mesh, contact_set_1: GeometrySet, contact_set_2=None, id=None
):
    """Adds two beam-to-beam contact boundary conditions to the given mesh and
    estimates automatically the id of them based on all previously added beam-
    to-beam contact boundary conditions of the mesh. If only contact_set_1 is
    specified, the given set will be automatically selected for both sets.

    Args:
        mesh (Mesh): Mesh to which the boundary conditions will be added.
        contact_set_1 (GeometrySet): GeometrySet 1 for contact boundary condition
        contact_set_2 (GeometrySet): GeometrySet 2 for contact boundary condition
        id (int): id of the two conditions

    Returns:
        id (int): used id of the two conditions.
    """

    # Copy contact set, if only one is provided
    if contact_set_2 is None:
        contact_set_2 = contact_set_1

    # Ensure that the GeometrySet is a line
    if (
        contact_set_1.geometry_type is not mpy.geo.line
        or contact_set_2.geometry_type is not mpy.geo.line
    ):
        raise ValueError(
            "Please specify the GeometrySet for the beam-to-beam contact as line set."
        )

    if id is None:
        id = get_next_possible_id_for_boundary_condition(
            mesh, mpy.bc.beam_to_beam_contact, r"COUPLING_ID (\d+)"
        )

    # Creates the two conditions with the same ID.
    for geometry_set in [contact_set_1, contact_set_2]:
        mesh.add(
            BoundaryCondition(
                geometry_set,
                "COUPLING_ID {}".format(id),
                bc_type=mpy.bc.beam_to_beam_contact,
            )
        )

    return id
