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
Create a couple of different mesh cases and test the performance.
"""


# Python imports.
import time
import numpy as np
import os
import socket
import warnings
import sys

# Meshpy imports.
from meshpy import (
    mpy,
    InputFile,
    Mesh,
    MaterialReissner,
    Beam3rHerm2Line3,
    Rotation,
    find_close_nodes,
)

from meshpy.mesh_creation_functions.beam_basic_geometry import create_beam_mesh_line

from tests.testing_utility import empty_testing_directory, testing_temp

# Cubitpy imports.
from cubitpy import cupy, CubitPy


def create_solid_block(file_path, nx, ny, nz):
    """
    Create a solid block (1 x 1 x 1) with (nx * ny * nz) elements.
    """

    # Initialize cubit.
    cubit = CubitPy()

    # Create brick.
    brick = cubit.brick(1)

    # Set mesh parameters.
    mesh_size = [
        [nx, [2, 4, 6, 8]],
        [ny, [1, 3, 5, 7]],
        [nz, [9, 10, 11, 12]],
    ]
    for [n_el, curves] in mesh_size:
        for i in curves:
            cubit.set_line_interval(brick.curves()[i - 1], n_el)
    brick.volumes()[0].mesh()

    # Add block and sets.
    cubit.add_element_type(
        brick.volumes()[0],
        cupy.element_type.hex8,
        name="brick",
        bc_description="MAT 1 KINEM nonlinear EAS none",
    )
    counter = 0
    for item in brick.vertices():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.neumann,
            bc_description="NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0",
        )
        counter += 1
    for item in brick.curves():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.dirichlet,
            bc_description="NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0",
        )
        counter += 1
    for item in brick.surfaces():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.neumann,
            bc_description="NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0",
        )
        counter += 1

    # Export mesh
    cubit.create_dat(file_path)


def load_solid(solid_file, full_import):
    """
    Load a solid into an input file.
    """

    mpy.set_default_values()
    mpy.import_mesh_full = full_import
    InputFile(dat_file=solid_file)


def create_large_beam_mesh(nx, ny, nz, n_el):
    """
    Create a beam grid on the domain (1 x 1 x 1) with (nx * ny * nz)
    "grid cells".
    """

    mesh = InputFile()
    material = MaterialReissner(radius=0.25 / np.max([nx, ny, nz]))

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [ix / nx, iy / ny, 0],
                [ix / nx, iy / ny, 1],
                n_el=nz * n_el,
            )
    for iy in range(ny + 1):
        for iz in range(nz + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [0, iy / ny, iz / nz],
                [1, iy / ny, iz / nz],
                n_el=nx * n_el,
            )
    for iz in range(nz + 1):
        for ix in range(nx + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [ix / nx, 0, iz / nz],
                [ix / nx, 1, iz / nz],
                n_el=ny * n_el,
            )
    return mesh


class TestPerformance(object):
    """
    A class to test meshpy performance.
    """

    # Set expected test times.
    expected_times = {}
    expected_times["adonis"] = {
        "cubitpy_create_solid": 3.2,
        "meshpy_load_solid": 1.1,
        "meshpy_load_solid_full": 3.5,
        "meshpy_create_beams": 7.2,
        "meshpy_rotate": 0.6,
        "meshpy_translate": 0.6,
        "meshpy_reflect": 0.7,
        "meshpy_wrap_around_cylinder": 3.0,
        "meshpy_wrap_around_cylinder_without_check": 0.9,
        "meshpy_find_close_nodes": 2.0,
        "meshpy_write_dat": 12.5,
        "meshpy_write_vtk": 19,
    }
    expected_times["ares.bauv.unibw-muenchen.de"] = {
        "cubitpy_create_solid": 3.2,
        "meshpy_load_solid": 0.9,
        "meshpy_load_solid_full": 3.0,
        "meshpy_create_beams": 7.2,
        "meshpy_rotate": 0.6,
        "meshpy_translate": 0.5,
        "meshpy_reflect": 0.7,
        "meshpy_wrap_around_cylinder": 3.0,
        "meshpy_wrap_around_cylinder_without_check": 0.7,
        "meshpy_find_close_nodes": 2.0,
        "meshpy_write_dat": 13.0,
        "meshpy_write_vtk": 24.0,
    }
    expected_times["sisyphos.bauv.unibw-muenchen.de"] = {
        "cubitpy_create_solid": 3.0,
        "meshpy_load_solid": 0.9,
        "meshpy_load_solid_full": 2.8,
        "meshpy_create_beams": 6.0,
        "meshpy_rotate": 0.6,
        "meshpy_translate": 0.5,
        "meshpy_reflect": 0.7,
        "meshpy_wrap_around_cylinder": 2.0,
        "meshpy_wrap_around_cylinder_without_check": 0.7,
        "meshpy_find_close_nodes": 1.6,
        "meshpy_write_dat": 10.5,
        "meshpy_write_vtk": 17.0,
    }

    def __init__(self):
        """
        Initialize counters.
        """

        self.passed_tests = 0
        self.failed_tests = 0

    def time_function(self, name, funct, args=None, kwargs=None):
        """
        Execute a function and check if the time is as expected.
        """

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        # Get the expected time for this function.
        host = socket.gethostname()
        if host in self.expected_times.keys():
            if name in self.expected_times[host].keys():
                expected_time = self.expected_times[host][name]
            else:
                raise ValueError("Function name {} not found!".format(name))
        else:
            raise ValueError("Host {} not found!".format(host))

        # Time before the execution.
        start_time = time.time()

        # Execute the function.
        return_val = funct(*args, **kwargs)

        # Check the elapsed time.
        elapsed_time = time.time() - start_time
        print(
            "Times for {}:\n".format(name)
            + "    Expected: {:.3g}sec\n".format(expected_time)
            + "    Actual:   {:.3g}sec".format(elapsed_time)
        )
        if expected_time > elapsed_time:
            self.passed_tests += 1
            print("    OK")
        else:
            self.failed_tests += 1
            print("    FAILED")
            warnings.warn("Expected time not reached in function {}!".format(name))

        # Return what the function would have given.
        return return_val


if __name__ == "__main__":
    """Execute part of script."""

    # Directories and files for testing.
    testing_solid_block = os.path.join(testing_temp, "performance_testing_solid.dat")
    testing_beam = os.path.join(testing_temp, "performance_testing_beam.dat")

    empty_testing_directory()

    test_performance = TestPerformance()

    test_performance.time_function(
        "cubitpy_create_solid",
        create_solid_block,
        args=[testing_solid_block, 100, 100, 10],
    )

    test_performance.time_function(
        "meshpy_load_solid", load_solid, args=[testing_solid_block, False]
    )

    test_performance.time_function(
        "meshpy_load_solid_full", load_solid, args=[testing_solid_block, True]
    )

    mesh = test_performance.time_function(
        "meshpy_create_beams", create_large_beam_mesh, args=[40, 40, 10, 2]
    )

    test_performance.time_function(
        "meshpy_rotate", Mesh.rotate, args=[mesh, Rotation([1, 1, 0], np.pi / 3)]
    )

    test_performance.time_function(
        "meshpy_translate", Mesh.translate, args=[mesh, [0.5, 0, 0]]
    )

    test_performance.time_function(
        "meshpy_reflect", Mesh.reflect, args=[mesh, [0.5, 0.4, 0.1]]
    )

    test_performance.time_function(
        "meshpy_wrap_around_cylinder",
        Mesh.wrap_around_cylinder,
        args=[mesh],
        kwargs={"radius": 1.0},
    )

    test_performance.time_function(
        "meshpy_wrap_around_cylinder_without_check",
        Mesh.wrap_around_cylinder,
        args=[mesh],
        kwargs={"radius": 1.0, "advanced_warning": False},
    )

    test_performance.time_function(
        "meshpy_find_close_nodes", find_close_nodes, args=[mesh.nodes]
    )

    test_performance.time_function(
        "meshpy_write_dat", InputFile.write_input_file, args=[mesh, testing_beam]
    )

    test_performance.time_function(
        "meshpy_write_vtk",
        Mesh.write_vtk,
        args=[mesh],
        kwargs={
            "output_name": "performance_testing_beam",
            "output_directory": testing_temp,
        },
    )

    if test_performance.failed_tests > 0:
        sys.exit(1)
    else:
        sys.exit(0)
