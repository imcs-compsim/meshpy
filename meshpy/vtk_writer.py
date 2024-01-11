# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
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
# -----------------------------------------------------------------------------
"""
This module provides a class that is used to write VTK files.
"""

# Python modules.
import vtk
import numpy as np
import numbers
import os
import warnings

# Meshpy modules.
from .conf import mpy


def add_point_data_node_sets(point_data, nodes):
    """
    Add the information if a node is part of a set to the point_data vector
    for all nodes in the list 'nodes'.
    """

    # Get list with node set indices of the given nodes
    geometry_set_list = []
    for node in nodes:
        geometry_set_list.extend(node.node_sets_link)

    # Remove double entries of list.
    geometry_set_list = list(set(geometry_set_list))

    # Loop over the geometry sets.
    for geometry_set in geometry_set_list:
        # Check which nodes are connected to a geometry set.
        data_vector = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            if geometry_set in node.node_sets_link:
                data_vector[i] = 1
            else:
                data_vector[i] = mpy.vtk_nan_int

        # Get the name of the geometry type.
        if geometry_set.geometry_type is mpy.geo.point:
            geometry_name = "geometry_point"
        elif geometry_set.geometry_type is mpy.geo.line:
            geometry_name = "geometry_line"
        elif geometry_set.geometry_type is mpy.geo.surface:
            geometry_name = "geometry_surface"
        elif geometry_set.geometry_type is mpy.geo.volume:
            geometry_name = "geometry_volume"
        else:
            raise TypeError("The geometry type is wrong!")

        # Add the data vector.
        set_name = "{}_set_{}".format(
            geometry_name, mpy.vtk_node_set_format.format(geometry_set.n_global)
        )
        point_data[set_name] = (data_vector, mpy.vtk_type.int)


def _get_data_value_and_type(data):
    """
    Return the data and its type if one was given.
    The default type, if none was given is float.
    """
    if type(data) is tuple:
        return data[0], data[1]
    else:
        return data, mpy.vtk_type.float


def _get_vtk_array_type(data):
    """
    Return the corresponding meshpy type.
    """
    data_type = data.GetDataTypeAsString()
    if data_type == "int":
        return mpy.vtk_type.int
    elif data_type == "double":
        return mpy.vtk_type.float
    else:
        raise ValueError('Got unexpected type ""!'.format(data_type))


class VTKWriter(object):
    """A class that manages VTK cells and data and can also create them."""

    def __init__(self):
        # Initialize VTK objects.
        self.points = vtk.vtkPoints()
        self.grid = vtk.vtkUnstructuredGrid()

        # Link points to grid.
        self.grid.SetPoints(self.points)

        # Container for output data.
        self.data = {}
        for key1 in mpy.vtk_geo:
            for key2 in mpy.vtk_tensor:
                self.data[key1, key2] = {}

    def add_cell(
        self, cell_type, coordinates, topology=None, cell_data=None, point_data=None
    ):
        """
        Create a cell and add it to the global array.

        Args
        ----
        cell_type: VTK_type
            Type of cell that will be created.
        coordinates: [3d vector]
            Coordinated of points for this cell.
        topology: [int]
            The connectivity between the cell and the coordinates. If nothing
            is given, it is assumed that the coordinates are in the right order
            for the cell.
        cell_data, point_data: dic
            A dictionary containing data that will be added to this cell,
            either as cell data, or point data for each point of the cell.
            There are some checks in place, but the length of the data should
            be chosen carefully. If some cells do not have a filed, that field
            will be set to 0 for this cell / points.
        """

        # Consistency checks.
        # Size of coordinates and topology.
        n_points = len(coordinates)
        if topology is None:
            topology = list(range(n_points))
        else:
            if not n_points == len(topology):
                raise ValueError(
                    "Coordinates is of size {}, while topology is of size {}!".format(
                        n_points, len(topology)
                    )
                )

        # Check if point data containers are of the correct size.
        if point_data is not None:
            for key, item_value in point_data.items():
                value, _data_type = _get_data_value_and_type(item_value)
                if not len(value) == n_points:
                    raise IndexError(
                        (
                            "The length of coordinates is {},"
                            + "the length of {} is {}, does not match!"
                        ).format(n_points, key, len(value))
                    )

        # Check if data container already exists. If not, add it and also add
        # previous entries.
        for data_container, vtk_geom_type in [
            (input_data, vtk_geom_type)
            for (input_data, vtk_geom_type) in [
                (cell_data, mpy.vtk_geo.cell),
                (point_data, mpy.vtk_geo.point),
            ]
            if input_data is not None
        ]:
            # Loop through output fields.
            for key, item_value in data_container.items():
                # Get the data and the value type (int or float).
                value, data_type = _get_data_value_and_type(item_value)

                # Data type.
                if vtk_geom_type == mpy.vtk_geo.cell:
                    vtk_tensor_type = self._get_vtk_data_type(value)
                else:
                    for item in value:
                        vtk_tensor_type = self._get_vtk_data_type(item)

                # Check if key already exists.
                if key not in self.data[vtk_geom_type, vtk_tensor_type].keys():
                    # Set up the VTK data array.
                    if data_type is mpy.vtk_type.float:
                        data = vtk.vtkDoubleArray()
                    else:
                        data = vtk.vtkIntArray()
                    data.SetName(key)
                    if vtk_tensor_type == mpy.vtk_tensor.scalar:
                        data.SetNumberOfComponents(1)
                    else:
                        data.SetNumberOfComponents(3)

                    # Add the empty values for all previous cells / points.
                    if vtk_geom_type == mpy.vtk_geo.cell:
                        n_items = self.grid.GetNumberOfCells()
                    else:
                        n_items = self.grid.GetNumberOfPoints()
                    for i in range(n_items):
                        self._add_data(data, vtk_tensor_type)
                    self.data[vtk_geom_type, vtk_tensor_type][key] = data

                else:
                    # In this case we just check that the already existing
                    # data has the same type.
                    data_array = self.data[vtk_geom_type, vtk_tensor_type][key]
                    if not _get_vtk_array_type(data_array) == data_type:
                        raise ValueError(
                            (
                                'The existing data with the key "{}"'
                                + ' is of type "{}", but the type you tried to add'
                                + ' is "{}"!'
                            ).format(key, data_array.GetDataTypeAsString(), data_type)
                        )

        # Create the cell.
        geometry_item = cell_type()
        geometry_item.GetPointIds().SetNumberOfIds(n_points)

        # Create the connection between the coordinates.
        n_grid_points = self.points.GetNumberOfPoints()
        for i, coord in enumerate(coordinates):
            # Add the coordinate to the global list of coordinates.
            self.points.InsertNextPoint(coord[0], coord[1], coord[2])

            # Set the local connectivity.
            geometry_item.GetPointIds().SetId(i, n_grid_points + topology[i])

        # Add to global cells.
        self.grid.InsertNextCell(
            geometry_item.GetCellType(), geometry_item.GetPointIds()
        )

        # Add to global data. Loop over data items and check if there is
        # something to be added in this cell. If not an empty value is added.
        for [key_geom, key_data], data in self.data.items():
            # Get input data container.
            if key_geom == mpy.vtk_geo.cell:
                data_container = cell_data
            else:
                data_container = point_data
            if data_container is None:
                data_container = {}

            for key, value in data.items():
                # Check if an existing field is also given for this function.
                if key in data_container.keys():
                    # Get the data and the value type (int or float).
                    data_values, _ = _get_data_value_and_type(data_container[key])

                    # Add the given data.
                    if key_geom == mpy.vtk_geo.cell:
                        self._add_data(value, key_data, non_zero_data=data_values)
                    else:
                        for item in data_values:
                            self._add_data(value, key_data, non_zero_data=item)
                else:
                    # Add empty data.
                    if key_geom == mpy.vtk_geo.cell:
                        self._add_data(value, key_data)
                    else:
                        for item in range(n_points):
                            self._add_data(value, key_data)

    def _get_vtk_data_type(self, data):
        """Return the type of data. Check if data matches an expected case."""

        if isinstance(data, list) or isinstance(data, np.ndarray):
            if len(data) == 3:
                return mpy.vtk_tensor.vector
            else:
                raise IndexError(
                    "Only 3d vectors are implemented yet! Got "
                    + "len(data) = {}".format(len(data))
                )
        elif isinstance(data, numbers.Number):
            return mpy.vtk_tensor.scalar

        raise ValueError("Data {} did not match any expected case!".format(data))

    def _add_data(self, data, vtk_tensor_type, non_zero_data=None):
        """Add data to a VTK data array."""

        if _get_vtk_array_type(data) == mpy.vtk_type.int:
            nan_value = mpy.vtk_nan_int
        elif _get_vtk_array_type(data) == mpy.vtk_type.float:
            nan_value = mpy.vtk_nan_float

        if vtk_tensor_type == mpy.vtk_tensor.scalar:
            if non_zero_data is None:
                data.InsertNextTuple1(nan_value)
            else:
                data.InsertNextTuple1(non_zero_data)
        else:
            if non_zero_data is None:
                data.InsertNextTuple3(nan_value, nan_value, nan_value)
            else:
                data.InsertNextTuple3(
                    non_zero_data[0], non_zero_data[1], non_zero_data[2]
                )

    def complete_data(self):
        """Add the stored data to the vtk grid"""
        for (key_geom, _key_data), value in self.data.items():
            for vtk_data in value.values():
                if key_geom == mpy.vtk_geo.cell:
                    self.grid.GetCellData().AddArray(vtk_data)
                else:
                    self.grid.GetPointData().AddArray(vtk_data)

    def write_vtk(self, filepath, *, ascii=False):
        """Write the VTK geometry and data to a file.

        Args
        ----
        filepath: str
            Path to output file. The file extension should be vtu.
        ascii: bool
            If the data should be written in human readable text.
        """

        # Check if directory for file exits.
        if not os.path.isdir(os.path.dirname(filepath)):
            raise ValueError(
                "Directory {} does not exist!".format(os.path.dirname(filepath))
            )

        # Initialize VTK writer.
        writer = vtk.vtkXMLUnstructuredGridWriter()

        # Set the ascii flag.
        if ascii:
            writer.SetDataModeToAscii()

        # Check the file extension.
        _filename, file_extension = os.path.splitext(filepath)
        if not file_extension.lower() == ".vtu":
            warnings.warn(
                'The extension should be "vtu", got {}!'.format(file_extension)
            )

        # Write geometry and data to file.
        writer.SetFileName(filepath)
        writer.SetInputData(self.grid)
        writer.Write()
