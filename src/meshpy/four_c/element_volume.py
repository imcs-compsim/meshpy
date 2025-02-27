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
"""This file implements volume elements for 4C."""

from meshpy.core.element_volume import VolumeElement


class SolidRigidSphere(VolumeElement):
    """A rigid sphere solid element."""

    def __init__(self, **kwargs):
        """Initialize solid sphere object."""
        VolumeElement.__init__(self, **kwargs)

        # Set radius of sphere from input file.
        arg_name = self.dat_post_nodes.split()[0]
        if not arg_name == "RADIUS":
            raise ValueError(
                "The first argument after the node should be "
                f'RADIUS, but it is "{arg_name}"!'
            )
        self.radius = float(self.dat_post_nodes.split()[1])
