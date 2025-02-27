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
"""This file provides functions to create Abaqus beam element classes to be
used with MeshPy."""

import numpy as np

from meshpy.core.element_beam import Beam
from meshpy.core.material import MaterialBeam


def generate_abaqus_beam(beam_type: str):
    """Return a class representing a beam in Abaqus. This class can be used in
    the standard MeshPy mesh generation functions.

    Args
    ----
    beam_type: str:
        Abaqus identifier for this beam element. For more details, have a look
        at the Abaqus manual on "Choosing a beam element"
    """

    if not beam_type[0].lower() == "b":
        raise TypeError("Could not identify the given Abaqus beam element")

    n_dim = int(beam_type[1])
    element_type = int(beam_type[2])

    if not n_dim == 3:
        raise ValueError("Currently only 3D beams in Abaqus are supported")
    if element_type == 1:
        n_nodes = 2
    elif element_type == 2:
        n_nodes = 3
    elif element_type == 3:
        n_nodes = 2
    else:
        raise ValueError(f"Got unexpected element_type {element_type}")

    # Define the class variable responsible for creating the nodes.
    nodes_create = np.linspace(-1, 1, num=n_nodes)

    # Create the Abaqus beam class.
    return type(
        "BeamAbaqus" + beam_type,
        (Beam,),
        {
            "beam_type": beam_type,
            "nodes_create": nodes_create,
            "n_dim": n_dim,
        },
    )


class AbaqusBeamMaterial(MaterialBeam):
    """A class representing an Abaqus beam material."""

    def __init__(self, name: str):
        """Initialize the material. For now it is only supported to state the
        name of the resulting element set here. The material and cross-section
        lines in the input file have to be defined manually.

        Args
        ----
        name: str
            Name of the material, this will be the name of the resulting element set
        """
        super().__init__(data=name)
