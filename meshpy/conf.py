# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
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
This module defines a global object that manages all kind of stuff regarding
meshpy.
"""

# Python imports.
from enum import Enum, auto


class Geometry(Enum):
    """Enum for geometry types."""

    point = auto()
    line = auto()
    surface = auto()
    volume = auto()


class BoundaryCondition(Enum):
    """Enum for boundary condition types."""

    dirichlet = auto()
    neumann = auto()
    moment_euler_bernoulli = auto()
    beam_to_solid_volume_meshtying = auto()
    beam_to_solid_surface_meshtying = auto()
    beam_to_solid_surface_contact = auto()
    point_coupling = auto()
    point_coupling_penalty = auto()


class BeamType(Enum):
    """Enum for beam types."""

    reissner = auto()
    kirchhoff = auto()
    euler_bernoulli = auto()


class CouplingDofType(Enum):
    """Enum for coupling types."""

    fix = auto()
    joint = auto()


class BeamToSolidInteractionType(Enum):
    """Enum for beam-to-solid interaction types."""

    volume_meshtying = auto()
    surface_meshtying = auto()


class DoubleNodes(Enum):
    """Enum for handing double nodes in Neumann conditions."""

    remove = auto()
    keep = auto()


class GeometricSearchAlgorithm(Enum):
    """Enum for VTK value types."""

    automatic = auto()
    brute_force_cython = auto()
    binning_cython = auto()
    boundary_volume_hierarchy_arborx = auto()


class VTKGeometry(Enum):
    """Enum for VTK geometry types (for now cells and points)."""

    point = auto()
    cell = auto()


class VTKTensor(Enum):
    """Enum for VTK tensor types."""

    scalar = auto()
    vector = auto()


class VTKType(Enum):
    """Enum for VTK value types."""

    int = auto()
    float = auto()


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

        # Beam-to-solid interaction types.
        self.beam_to_solid = BeamToSolidInteractionType

        # Coupling types.
        self.coupling_dof = CouplingDofType

        # Handling of multiple nodes in Neumann bcs.
        self.double_nodes = DoubleNodes

        # Geometric search options.
        self.geometric_search_algorithm = GeometricSearchAlgorithm

        # VTK types.
        # Geometry types, cell or point.
        self.vtk_geo = VTKGeometry
        # Tensor types, scalar or vector.
        self.vtk_tensor = VTKTensor
        # Data types, integer or float.
        self.vtk_type = VTKType

    def set_default_values(self):

        # Version information.
        self.git_sha = None
        self.git_date = None

        # Precision for floats in output.
        self.dat_precision = "{:.12g}"

        # Set the epsilons for comparison of different types of values.
        self.eps_quaternion = 1e-10
        self.eps_pos = 1e-10
        self.eps_knot_vector = 1e-10

        # Allow the rotation of beams when connected and the triads do not
        # match.
        self.allow_beam_rotation = True

        # Geometric search options.
        self.geometric_search_max_nodes_brute_force = 1000
        self.geometric_search_binning_n_bin = 10

        # Values for the formating of the input file.
        self.dat_len_section = 80

        # Import meshes as pure dat or import the geometry.
        self.import_mesh_full = False

        # Number of digits for node set output (this will be set in the
        # Mesh.get_unique_geometry_sets() method.
        self.vtk_node_set_format = "{:05}"
        # Nan values for vtk data, since we currently can't set nan explicitly.
        self.vtk_nan_int = -1
        self.vtk_nan_float = 0.0

        # Check for overlapping elements when creating a dat file.
        self.check_overlapping_elements = True

        # Lines to be added to each created input file
        self.input_file_meshpy_header = [
            "-" * 77,
            "This input file was created with MeshPy.",
            "Copyright (c) 2021 Ivo Steinbrecher",
            "           Institute for Mathematics and Computer-Based Simulation",
            "           Universitaet der Bundeswehr Muenchen",
            "           https://www.unibw.de/imcs-en",
            "-" * 77,
        ]


mpy = MeshPy()
