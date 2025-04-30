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
"""Create a couple of different mesh cases and test the performance."""

import time
import warnings

import numpy as np
import pytest

from meshpy.core.conf import mpy
from meshpy.core.mesh import Mesh
from meshpy.core.rotation import Rotation
from meshpy.four_c.element_beam import Beam3rHerm2Line3
from meshpy.four_c.input_file import InputFile
from meshpy.four_c.material import MaterialReissner
from meshpy.geometric_search.find_close_points import (
    FindClosePointAlgorithm,
    find_close_points,
)
from meshpy.mesh_creation_functions.beam_basic_geometry import create_beam_mesh_line
from meshpy.utils.environment import cubitpy_is_available
from meshpy.utils.nodes import find_close_nodes

if cubitpy_is_available():
    from cubitpy import CubitPy, cupy


def create_solid_block(file_path, nx, ny, nz):
    """Create a solid block (1 x 1 x 1) with (nx * ny * nz) elements."""

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
    for item in brick.volumes():
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
    """Load a solid into an input file."""

    mpy.import_mesh_full = full_import
    InputFile(yaml_file=solid_file)


def create_large_beam_mesh(n_x, n_y, n_z, n_el):
    """Create a beam grid on the domain (1 x 1 x 1) with (nx * ny * nz) "grid
    cells"."""

    mesh = Mesh()
    material = MaterialReissner(radius=0.25 / np.max([n_x, n_y, n_z]))

    for i_x in range(n_x + 1):
        for i_y in range(n_y + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [i_x / n_x, i_y / n_y, 0],
                [i_x / n_x, i_y / n_y, 1],
                n_el=n_z * n_el,
            )
    for i_y in range(n_y + 1):
        for i_z in range(n_z + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [0, i_y / n_y, i_z / n_z],
                [1, i_y / n_y, i_z / n_z],
                n_el=n_x * n_el,
            )
    for i_z in range(n_z + 1):
        for i_x in range(n_x + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [i_x / n_x, 0, i_z / n_z],
                [i_x / n_x, 1, i_z / n_z],
                n_el=n_y * n_el,
            )
    return mesh


class PerformanceTest(object):
    """A class to test meshpy performance."""

    def __init__(self, expected_times):
        """Initialize counters."""

        self.expected_times = expected_times
        self.passed_tests = 0
        self.failed_tests = 0

    def time_function(self, name, funct, args=None, kwargs=None):
        """Execute a function and check if the time is as expected."""

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        # Get the expected time for this function.
        if name in self.expected_times.keys():
            expected_time = self.expected_times[name]
        else:
            raise ValueError("Function name {} not found!".format(name))

        # Time before the execution.
        start_time = time.time()

        # Execute the function.
        return_val = funct(*args, **kwargs)

        # Check the elapsed time.
        elapsed_time = time.time() - start_time
        print(
            f"Times for {name}:\n"
            f"    Expected: {expected_time:.3g}sec\n"
            f"    Actual:   {elapsed_time:.3g}sec"
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


def get_geometric_search_time(algorithm, n_points, n_runs):
    """Return the time needed to perform geometric search functions."""

    np.random.seed(seed=1)
    points = np.random.rand(n_points, 3)

    start = time.time()
    for i in range(n_runs):
        find_close_points(points, algorithm=algorithm)
    return time.time() - start


@pytest.mark.skip(
    reason="Temporarily disabled due to switch to .yaml based input files - check if test is necessary and fix"
)
@pytest.mark.performance
def test_performance(tmp_path):
    """The actual performance test."""

    # Directories and files for testing.
    testing_solid_block = tmp_path / "performance_testing_solid.4C.yaml"
    testing_beam = tmp_path / "performance_testing_beam.4C.yaml"

    # These are the expected test times that should not be exceeded
    expected_times = {
        "cubitpy_create_solid": 8.0,
        "meshpy_load_solid": 1.5,
        "meshpy_load_solid_full": 3.5,
        "meshpy_create_beams": 9.0,
        "meshpy_rotate": 0.6,
        "meshpy_translate": 0.5,
        "meshpy_reflect": 0.7,
        "meshpy_wrap_around_cylinder": 2.0,
        "meshpy_wrap_around_cylinder_without_check": 0.7,
        "meshpy_find_close_nodes": 0.5,
        "meshpy_write_dat": 9.0,
        "meshpy_write_vtk": 4.5,
        "meshpy_write_vtk_smooth": 9.0,
        "geometric_search_find_nodes_brute_force": 0.05,
    }
    test_performance = PerformanceTest(expected_times)

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

    # Use a smaller mesh for shorter times in vtk testing
    mesh_smaller_for_output = create_large_beam_mesh(20, 20, 10, 2)

    test_performance.time_function(
        "meshpy_write_vtk",
        Mesh.write_vtk,
        args=[mesh_smaller_for_output],
        kwargs={
            "output_name": "performance_testing_beam",
            "output_directory": tmp_path,
            "beam_centerline_visualization_segments": 1,
        },
    )

    test_performance.time_function(
        "meshpy_write_vtk_smooth",
        Mesh.write_vtk,
        args=[mesh_smaller_for_output],
        kwargs={
            "output_name": "performance_testing_beam",
            "output_directory": tmp_path,
            "beam_centerline_visualization_segments": 5,
        },
    )

    test_performance.time_function(
        "geometric_search_find_nodes_brute_force",
        get_geometric_search_time,
        args=[FindClosePointAlgorithm.brute_force_cython, 100, 1000],
    )

    assert test_performance.failed_tests == 0
