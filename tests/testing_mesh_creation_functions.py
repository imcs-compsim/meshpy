# -*- coding: utf-8 -*-
"""
This script is used to test the mesh creation functions.
"""

# Python imports.
import unittest
import numpy as np
import os

# Meshpy imports.
from meshpy import (mpy, Rotation, InputFile, MaterialReissner, MaterialBeam,
    BoundaryCondition, Node, BaseMeshItem, VTKWriter, compare_xml, Mesh,
    get_close_nodes, GeometryName, GeometrySet, MaterialKirchhoff, Beam3k,
    flatten, Beam, Coupling, Beam3rHerm2Lin3, Function)

# Geometry functions.
from meshpy.mesh_creation_functions.beam_stent import create_beam_mesh_stent

# Testing imports.
from tests.testing_utility import (testing_temp, testing_input,
    compare_strings)


class TestMeshCreationFunctions(unittest.TestCase):
    """
    Test the mesh creation functions.
    """

    def test_stent(self):
        """
        Test the stent creation function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Add material and function.
        mat = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Create mesh.
        create_beam_mesh_stent(input_file, Beam3rHerm2Lin3, mat, 20, 2, 10, 8,
            2)
        
        
        input_file.write_vtk('stent', '/home/ivo/temp')
        
        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_meshpy_stent_reference.dat')
        compare_strings(
            self,
            'test_meshpy_stent',
            ref_file,
            input_file.get_string(header=False))


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
