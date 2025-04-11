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
"""This module implements the class that will be used as the base for all items
that are in a mesh."""


class BaseMeshItem:
    """Base class for all objects that are related to a mesh."""

    def __init__(self, data=None, comments=None):
        """Create the object.

        Args
        ----
        data: str, list(str)
            Data for this object.
        TODO: comments
        """

        self.data = data

        # Overall index of this item in the mesh.
        self.i_global = None

        # Comments regarding this Mesh Item.
        if comments is None:
            self.comments = []
        else:
            self.comments = comments


class BaseMeshItemFull(BaseMeshItem):
    """Base class for all objects that are related to a mesh and are fully
    created in MeshPy."""


class BaseMeshItemString(BaseMeshItem):
    """Base class for all objects that are imported from an input file as a
    plain string."""
