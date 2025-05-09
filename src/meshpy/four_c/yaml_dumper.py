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
"""Utility stuff to dump data to a yaml file."""

import numpy as _np
import yaml as _yaml

from meshpy.core.function import Function as _Function


class MeshPyDumper(_yaml.SafeDumper):
    """Dumper for MeshPy input files."""

    def ignore_aliases(self, data):
        """Don't alias shared objects."""
        return True


def function_representer(dumper, data: _Function):
    """Define how the Function object should be dumped.

    Since the only time we have the function object in the list to dump
    is when it is referred to in a BC, we only have to return the global
    index here.
    """
    if data.i_global is None:
        raise IndexError("The function does not have a global index!")
    return dumper.represent_int(data.i_global)


MeshPyDumper.add_representer(_Function, function_representer)


def numpy_float_representer(dumper, value):
    """Converter for numpy float to yaml."""
    return dumper.represent_float(float(value))


def numpy_int_representer(dumper, value):
    """Converter for numpy int to yaml."""
    return dumper.represent_int(int(value))


def numpy_bool_representer(dumper, value):
    """Converter for numpy bool to yaml."""
    return dumper.represent_bool(bool(value))


MeshPyDumper.add_representer(_np.float32, numpy_float_representer)
MeshPyDumper.add_representer(_np.float64, numpy_float_representer)
MeshPyDumper.add_representer(_np.int32, numpy_int_representer)
MeshPyDumper.add_representer(_np.int64, numpy_int_representer)
MeshPyDumper.add_representer(_np.bool_, numpy_bool_representer)
