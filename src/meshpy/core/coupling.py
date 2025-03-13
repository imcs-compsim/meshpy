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
"""This module implements a class to couple geometry together."""

import numpy as np

from meshpy.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from meshpy.core.conf import mpy as _mpy
from meshpy.core.geometry_set import GeometrySet as _GeometrySet
from meshpy.core.geometry_set import GeometrySetBase as _GeometrySetBase


class Coupling(_BoundaryConditionBase):
    """Represents a coupling between geometries in 4C."""

    def __init__(
        self,
        geometry,
        coupling_type,
        coupling_dof_type,
        *,
        check_overlapping_nodes=True,
        check_at_init=True,
        **kwargs,
    ):
        """Initialize this object.

        Args
        ----
        geometry: GeometrySet, [Nodes], int
            Geometry that this boundary condition acts on.
        coupling_type: mpy.bc
            Type of point coupling.
        coupling_dof_type: mpy.coupling_dof, str
            If this is a string it is the string that will be used in the input
            file, otherwise it has to be of type mpy.coupling_dof.
        check_overlapping_nodes: bool
            If all nodes of this coupling condition have to be at the same
            physical position.
        check_at_init: bool
            If the previous check should be performed at initialization. This is required
            when importing a coupling from a dat file as the nodes themselves are not build
            up when the coupling is read.
        """

        if isinstance(geometry, int):
            # This is the case if the boundary condition is read from an existing dat file
            pass
        elif isinstance(geometry, _GeometrySetBase):
            pass
        elif isinstance(geometry, list):
            geometry = _GeometrySet(geometry)
        else:
            raise TypeError(
                f"Coupling expects a GeometrySetBase item, got {type(geometry)}"
            )

        # Couplings only work for point sets
        if (
            isinstance(geometry, _GeometrySetBase)
            and geometry.geometry_type is not _mpy.geo.point
        ):
            raise TypeError("Couplings are only implemented for point sets.")

        super().__init__(geometry, bc_type=coupling_type, **kwargs)
        self.coupling_dof_type = coupling_dof_type
        self.check_overlapping_nodes = check_overlapping_nodes

        if check_at_init:
            # Perform the checks on this boundary condition
            self.check()

    def check(self):
        """Check that all nodes that are coupled have the same position
        (depending on the check_overlapping_nodes parameter)."""

        if not self.check_overlapping_nodes:
            return

        nodes = self.geometry_set.get_points()
        diff = np.zeros([len(nodes), 3])
        for i, node in enumerate(nodes):
            # Get the difference to the first node
            diff[i, :] = node.coordinates - nodes[0].coordinates
        if np.max(np.linalg.norm(diff, axis=1)) > _mpy.eps_pos:
            raise ValueError(
                "The nodes given to Coupling do not have the same position."
            )

    def _get_dat(self):
        """Return the dat line for this object.

        If no explicit string was given, it depends on the coupling type
        as well as the beam type.
        """

        if isinstance(self.coupling_dof_type, str):
            string = self.coupling_dof_type
        else:
            # In this case we have to check which beams are connected to the node.
            # TODO: Coupling also makes sense for different beam types, this can
            # be implemented at some point.
            nodes = self.geometry_set.get_points()
            beam_type = nodes[0].element_link[0].beam_type
            for node in nodes:
                for element in node.element_link:
                    if beam_type is not element.beam_type:
                        raise ValueError(
                            f'The first element in this coupling is of the type "{beam_type}" '
                            f'another one is of type "{element.beam_type}"! They have to be '
                            "of the same kind."
                        )
                    if beam_type is _mpy.beam.kirchhoff and element.rotvec is False:
                        raise ValueError(
                            "Couplings for Kirchhoff beams and rotvec==False not yet implemented."
                        )

            # In 4C it is not possible to couple beams of the same type, but
            # with different centerline discretizations, e.g. Beam3rHerm2Line3
            # and Beam3rLine2Line2, therefore, we check that all beams are
            # exactly the same type and discretization.
            # TODO: Remove this check once it is possible to couple beams, but
            # then also the syntax in the next few lines has to be adapted.
            beam_four_c_type = type(nodes[0].element_link[0])
            for node in nodes:
                for element in node.element_link:
                    if beam_four_c_type is not type(element):
                        raise ValueError(
                            "Coupling beams of different types is not yet possible!"
                        )

            string = beam_four_c_type.get_coupling_string(self.coupling_dof_type)

        return f"E {self.geometry_set.i_global} {string}"


def coupling_factory(geometry, coupling_type, coupling_dof_type, **kwargs):
    """Create coupling conditions for the nodes in geometry.

    Some solvers only allow coupling conditions containing two points at
    once, in that case we have to create multiple coupling conditions
    between the individual points to ensure the correct representation
    of the coupling.
    """

    if coupling_type is _mpy.bc.point_coupling_penalty:
        # Penalty point couplings in 4C can only contain two nodes. In this case
        # we expect the given geometry to be a list of nodes.
        main_node = geometry[0]
        return [
            Coupling([main_node, node], coupling_type, coupling_dof_type, **kwargs)
            for node in geometry[1:]
        ]
    else:
        return [Coupling(geometry, coupling_type, coupling_dof_type, **kwargs)]
