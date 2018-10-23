# -*- coding: utf-8 -*-
"""
This module defines a global object that manages all kind of stuff regarding
meshpy.
"""


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """

    def __init__(self):
        self.set_default_values()

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

        # Geometry types.
        self.point = 'geometry_point'
        self.line = 'geometry_line'
        self.surface = 'geometry_surface'
        self.volume = 'geometry_volume'
        self.geometry = [self.point, self.line, self.surface, self.volume]

        # Boundary conditions types.
        self.dirichlet = 'boundary_condition_dirichlet'
        self.neumann = 'boundary_condition_neumann'
        self.boundary_condition = [self.dirichlet, self.neumann]

        # Coupling types.
        self.coupling_fix = 'coupling_fix'
        self.coupling_fix_reuse = 'coupling_fix_reuse_nodes'
        self.coupling_joint = 'coupling_joint'

        # Handling of multiple nodes in neuman bcs.
        self.double_nodes_remove = 'double_nodes_remove'
        self.double_nodes_keep = 'double_nodes_keep'

        # VTK types.
        # Geometry types, cell or point.
        self.vtk_cell = 'vtk_cell'
        self.vtk_point = 'vtk_point'
        self.vtk_geom_types = [self.vtk_cell, self.vtk_point]
        # Data types, scalar or vector.
        self.vtk_scalar = 'vtk_scalar'
        self.vtk_vector = 'vtk_vector'
        self.vtk_data_types = [self.vtk_scalar, self.vtk_vector]
        # Number of digits for node set output (this will be set in the
        # Mesh.get_unique_geometry_sets() method.
        self.vtk_node_set_format = '{:05}'


mpy = MeshPy()
