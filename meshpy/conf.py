# -*- coding: utf-8 -*-
"""
This module defines a global object that manages all kind of stuff regarding
meshpy.
"""

# Python imports.
from enum import IntEnum


class Geometry(IntEnum):
    """Enum for geometry types."""
    point = 1
    line = 2
    surface = 3
    volume = 4


class BoundaryCondition(IntEnum):
    """Enum for boundary condition types."""
    dirichlet = 1
    neumann = 2
    moment_euler_bernoulli = 3


class BeamType(IntEnum):
    """Enum for beam types."""
    reissner = 1
    kirchhoff = 2
    bernoulli_euler = 3


class CouplingType(IntEnum):
    """Enum for coupling types."""
    fix = 1
    fix_reuse = 2
    joint = 3


class DoubleNodes(IntEnum):
    """Enum for handing double nodes in Neumann conditions."""
    remove = 1
    keep = 2


class VTKGeometry(IntEnum):
    """Enum for VTK geometry types (for now cells and points)."""
    point = 1
    cell = 2


class VTKData(IntEnum):
    """Enum for VTK data types."""
    scalar = 1
    vector = 2


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """

    def __init__(self):
        self.set_default_values()

        # Geometry types.
        self.geo = Geometry

        # Boundary conditions types.
        self.bc = BoundaryCondition

        # Beam types.
        self.beam = BeamType

        # Coupling types.
        self.coupling = CouplingType

        # Handling of multiple nodes in Neumann bcs.
        self.double_nodes = DoubleNodes

        # VTK types.
        # Geometry types, cell or point.
        self.vtk_geo = VTKGeometry
        # Data types, scalar or vector.
        self.vtk_data = VTKData

    def set_default_values(self):

        # Version information.
        self.version = '0.0.5'
        self.git_sha = None
        self.git_date = None

        # Precision for floats in output.
        self.dat_precision = '{:.12g}'

        # Set the epsilons for comparison of different types of values.
        self.eps_quaternion = 1e-10
        self.eps_pos = 1e-10

        # Allow the rotation of beams when connected and the triads do not
        # match.
        self.allow_beam_rotation = True

        # Binning options.
        self.binning = True
        self.binning_max_nodes_brute_force = 1000
        self.binning_n_bin = 10

        # Values for the formating of the input file.
        self.dat_len_section = 80

        # Import meshes as pure dat or import the geometry.
        self.import_mesh_full = False

        # Number of digits for node set output (this will be set in the
        # Mesh.get_unique_geometry_sets() method.
        self.vtk_node_set_format = '{:05}'

        # Check for overlapping elements when creating a dat file.
        self.check_overlapping_elements = True


mpy = MeshPy()
