# -*- coding: utf-8 -*-
"""
This module implements a class to handle boundary conditions in the input file.
"""

# Python modules.
import warnings

# Meshpy modules.
from .conf import mpy
from .base_mesh_item import BaseMeshItem
from .utility import get_close_nodes


class BoundaryConditionBase(BaseMeshItem):
    """
    This is a base object, which represents one boundary condition in the input
    file.
    """

    def __init__(self, geometry_set, bc_string, **kwargs):
        """
        Initialize the object.

        Args
        ----
        geometry_set: GeometrySet
            Geometry that this boundary condition acts on.
        bc_string: str
            Text that will be displayed in the input file for this boundary
            condition.
        """

        BaseMeshItem.__init__(self, is_dat=False, **kwargs)
        self.bc_string = bc_string
        self.geometry_set = geometry_set

    @classmethod
    def from_dat(cls, bc_key, line, **kwargs):
        """
        Get a boundary condition from an input line in a dat file. The geometry
        set is passed as integer (0 based index) and will be connected after
        the whole input file is parsed.
        """

        # Split up the input line.
        split = line.split()

        if (bc_key == mpy.bc.dirichlet
                or bc_key == mpy.bc.neumann
                or bc_key == mpy.bc.beam_to_solid_surface_meshtying
                or bc_key == mpy.bc.beam_to_solid_volume_meshtying):
            # Normal boundary condition (including beam-to-solid conditions).
            return BoundaryCondition(
                int(split[1]) - 1, ' '.join(split[3:]),
                bc_type=bc_key, **kwargs
                )
        else:
            raise ValueError('Got unexpected boundary condition!')


class BoundaryCondition(BoundaryConditionBase):
    """
    This object represents a Dirichlet, Neumann or beam-to-solid boundary
    condition.
    """

    def __init__(self, geometry_set, bc_string, format_replacement=None,
            bc_type=None, double_nodes=None, **kwargs):
        """
        Initialize the object.

        Args
        ----
        geometry_set: GeometrySet
            Geometry that this boundary condition acts on.
        bc_string: str
            Text that will be displayed in the input file for this boundary
            condition.
        format_replacement: str, list
            Replacement with the str.format() function for bc_string.
        bc_type: mpy.boundary
            Type of the boundary condition.
        double_nodes: mpy.double_nodes
            Depending on this parameter, it will be checked if point Neumann
            conditions do contain nodes at the same spatial positions.
        """

        BoundaryConditionBase.__init__(self, geometry_set, bc_string, **kwargs)
        self.bc_type = bc_type
        self.format_replacement = format_replacement

        # Check the parameters for this object.
        self._check_multiple_nodes(double_nodes=double_nodes)

    def _get_dat(self):
        """
        Add the content of this object to the list of lines.

        Args:
        ----
        lines: list(str)
            The contents of this object will be added to the end of lines.
        """

        if self.format_replacement:
            dat_string = self.bc_string.format(*self.format_replacement)
        else:
            dat_string = self.bc_string

        return 'E {} - {}'.format(
            self.geometry_set.n_global,
            dat_string
            )

    def _check_multiple_nodes(self, double_nodes=None):
        """
        Check for point Neumann boundaries that there is not a double
        Node in the set.
        """

        if isinstance(self.geometry_set, int):
            # In the case of solid imports this is a integer at initialization.
            return

        if double_nodes is mpy.double_nodes.keep:
            return

        if (self.bc_type == mpy.bc.neumann
                and self.geometry_set.geometry_type == mpy.geo.point):
            partners = get_close_nodes(self.geometry_set.nodes)
            # Create a list with nodes that will not be kept in the set.
            double_node_list = []
            for node_list in partners:
                for i, node in enumerate(node_list):
                    if i > 0:
                        double_node_list.append(node)
            if (len(double_node_list) > 0 and
                    double_nodes is mpy.double_nodes.remove):
                # Create the nodes again for the set.
                self.geometry_set.nodes = [node for node in
                    self.geometry_set.nodes if (node not in double_node_list)]
            elif len(double_node_list) > 0:
                warnings.warn('There are overlapping nodes in this point '
                    + 'Neumann boundary, and it is not specified on how to '
                    + 'handle them!')
