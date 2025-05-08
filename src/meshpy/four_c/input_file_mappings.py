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
"""This file provides the mappings between MeshPy objects and 4C input
files."""

from meshpy.core.conf import mpy as _mpy

# Define the names of sections and boundary conditions in the input file.
geometry_set_names = {
    _mpy.geo.point: "DNODE-NODE TOPOLOGY",
    _mpy.geo.line: "DLINE-NODE TOPOLOGY",
    _mpy.geo.surface: "DSURF-NODE TOPOLOGY",
    _mpy.geo.volume: "DVOL-NODE TOPOLOGY",
}
boundary_condition_names = {
    (_mpy.bc.dirichlet, _mpy.geo.point): "DESIGN POINT DIRICH CONDITIONS",
    (_mpy.bc.dirichlet, _mpy.geo.line): "DESIGN LINE DIRICH CONDITIONS",
    (_mpy.bc.dirichlet, _mpy.geo.surface): "DESIGN SURF DIRICH CONDITIONS",
    (_mpy.bc.dirichlet, _mpy.geo.volume): "DESIGN VOL DIRICH CONDITIONS",
    (_mpy.bc.locsys, _mpy.geo.point): "DESIGN POINT LOCSYS CONDITIONS",
    (_mpy.bc.locsys, _mpy.geo.line): "DESIGN LINE LOCSYS CONDITIONS",
    (_mpy.bc.locsys, _mpy.geo.surface): "DESIGN SURF LOCSYS CONDITIONS",
    (_mpy.bc.locsys, _mpy.geo.volume): "DESIGN VOL LOCSYS CONDITIONS",
    (_mpy.bc.neumann, _mpy.geo.point): "DESIGN POINT NEUMANN CONDITIONS",
    (_mpy.bc.neumann, _mpy.geo.line): "DESIGN LINE NEUMANN CONDITIONS",
    (_mpy.bc.neumann, _mpy.geo.surface): "DESIGN SURF NEUMANN CONDITIONS",
    (_mpy.bc.neumann, _mpy.geo.volume): "DESIGN VOL NEUMANN CONDITIONS",
    (
        _mpy.bc.moment_euler_bernoulli,
        _mpy.geo.point,
    ): "DESIGN POINT MOMENT EB CONDITIONS",
    (
        _mpy.bc.beam_to_solid_volume_meshtying,
        _mpy.geo.line,
    ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE",
    (
        _mpy.bc.beam_to_solid_volume_meshtying,
        _mpy.geo.volume,
    ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME",
    (
        _mpy.bc.beam_to_solid_surface_meshtying,
        _mpy.geo.line,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING LINE",
    (
        _mpy.bc.beam_to_solid_surface_meshtying,
        _mpy.geo.surface,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE",
    (
        _mpy.bc.beam_to_solid_surface_contact,
        _mpy.geo.line,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT LINE",
    (
        _mpy.bc.beam_to_solid_surface_contact,
        _mpy.geo.surface,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE",
    (_mpy.bc.point_coupling, _mpy.geo.point): "DESIGN POINT COUPLING CONDITIONS",
    (
        _mpy.bc.beam_to_beam_contact,
        _mpy.geo.line,
    ): "BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS",
    (
        _mpy.bc.point_coupling_penalty,
        _mpy.geo.point,
    ): "DESIGN POINT PENALTY COUPLING CONDITIONS",
    (
        "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
        _mpy.geo.surface,
    ): "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
}
