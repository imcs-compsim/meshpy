# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This module provides the simulation manager.
"""

from .simulation_manager import (Simulation, SimulationManager,
    wait_for_jobs_to_finish)

# Define the items that will be exported by default.
__all__ = [
    'Simulation',
    'SimulationManager',
    'wait_for_jobs_to_finish'
    ]
