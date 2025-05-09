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
"""This file contains the wrapper for the LocSys condition for 4c."""

from typing import List as _List
from typing import Optional as _Optional
from typing import Union as _Union

from meshpy.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from meshpy.core.conf import mpy as _mpy
from meshpy.core.function import Function as _Function
from meshpy.core.geometry_set import GeometrySet as _GeometrySet
from meshpy.core.rotation import Rotation as _Rotation
from meshpy.four_c.function_utility import (
    ensure_length_of_function_array as _ensure_length_of_function_array,
)


class LocSysCondition(_BoundaryCondition):
    """This object represents a locsys condition in 4C.

    It allows to rotate the local coordinate system used to apply
    Dirichlet boundary conditions.
    """

    def __init__(
        self,
        geometry_set: _GeometrySet,
        rotation: _Rotation,
        *,
        function_array: _Optional[_List[_Union[_Function, int]]] = None,
        update_node_position: bool = False,
        use_consistent_node_normal: bool = False,
        **kwargs,
    ):
        """Initialize the object.

        Args:
            geometry_set: Geometry that this boundary condition acts on
            rotation: Object that represents the rotation of the coordinate system
            function_array: _List containing functions
            update_node_position: Flag to enable the updated node position
            use_consistent_node_normal: Flag to use a consistent node normal
        """

        # Validate provided function array.
        if function_array is None:
            function_array = [0, 0, 0]
        else:
            function_array = _ensure_length_of_function_array(function_array, 3)

        condition_dict = {
            "ROTANGLE": rotation.get_rotation_vector().tolist(),
            "FUNCT": function_array,
            "USEUPDATEDNODEPOS": int(update_node_position),
        }

        # Append the condition string with consistent normal type for line and surface geometry
        if (
            geometry_set.geometry_type is _mpy.geo.line
            or geometry_set.geometry_type is _mpy.geo.surface
        ):
            condition_dict["USECONSISTENTNODENORMAL"] = int(use_consistent_node_normal)
        elif use_consistent_node_normal:
            raise ValueError(
                "The keyword use_consistent_node_normal only works for line and surface geometries."
            )

        super().__init__(
            geometry_set, data=condition_dict, bc_type=_mpy.bc.locsys, **kwargs
        )
