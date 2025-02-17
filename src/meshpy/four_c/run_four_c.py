# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""Provide a function that allows to run 4C."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from meshpy.utils.environment import get_env_variable


def run_four_c(
    input_file,
    output_dir,
    *,
    four_c_exe=None,
    mpi_command=None,
    n_proc=None,
    output_name="xxx",
    restart_step=None,
    restart_from=None,
    log_to_console=False,
):
    """Run a 4C simulation and return the exit code of the run.

    This function looks into the environment variables for some parameters:
        "MESHPY_FOUR_C_EXE"
        "MESHPY_MPI_COMMAND"
        "MESHPY_MPI_NUM_PROC"
    If the corresponding keyword arguments are set, they overwrite the environment
    variable.

    Args
    ----
    input_file: str
        Path to the dat file on the filesystem
    output_dir: str
        Directory where the simulation should be performed (will be created if
        it does not exist)
    four_c_exe: str
        Optionally explicitly specify path to the 4C executable
    mpi_command: str
        Command to launch MPI, defaults to "mpirun"
    n_proc: int
        Number of process used with MPI, defaults to 1
    output_name: str
        Base name of the output files
    restart_step: int
        Time step to restart from
    restart_from: str
        Path to initial simulation (relative to output_dir)
    log_to_console: bool
        If the 4C simulation output should be shown in the console.

    Return
    ----
    return_code: int
        Return code of 4C run
    """

    # Fist get all needed parameters
    if four_c_exe is None:
        four_c_exe = get_env_variable("MESHPY_FOUR_C_EXE")
    if mpi_command is None:
        mpi_command = get_env_variable("MESHPY_MPI_COMMAND", default="mpirun")
    if n_proc is None:
        n_proc = get_env_variable("MESHPY_MPI_NUM_PROC", default="1")

    # Setup paths and actual command to run
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, output_name + ".log")
    error_file = os.path.join(output_dir, output_name + ".err")
    command = mpi_command.split(" ") + [
        "-np",
        str(n_proc),
        four_c_exe,
        input_file,
        output_name,
    ]
    if restart_step is None and restart_from is None:
        pass
    elif restart_step is not None and restart_from is not None:
        command.extend([f"restart={restart_step}", f"restartfrom={restart_from}"])
    else:
        raise ValueError(
            "Provide either both or no argument of [restart_step, restart_from]"
        )

    # Actually run the command
    with open(log_file, "w") as stdout_file, open(error_file, "w") as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=output_dir,
            text=True,
        )

        for stdout_line in process.stdout:
            if log_to_console:
                sys.stdout.write(stdout_line)
            stdout_file.write(stdout_line)

        for stderr_line in process.stderr:
            if log_to_console:
                sys.stderr.write(stderr_line)
            stderr_file.write(stderr_line)

        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()
    return return_code


def clean_simulation_directory(sim_dir, *, ask_before_clean=False):
    """Clear the simulation directory. If it does not exist, it is created.
    Optionally the user can be asked before a deletion of files.

    Args
    ----
    sim_dir:
        Path to a directory
    ask_before_clean: bool
        Flag which indicates whether the user must confirm removal of files and directories
    """

    # Check if simulation directory exists.
    if os.path.exists(sim_dir):
        if ask_before_clean:
            print(f'Path "{sim_dir}" already exists')
        while True:
            if ask_before_clean:
                answer = input("DELETE all contents? (y/n): ")
            else:
                answer = "y"
            if answer.lower() == "y":
                for filename in os.listdir(sim_dir):
                    file_path = os.path.join(sim_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        raise ValueError(f"Failed to delete {file_path}. Reason: {e}")
                return
            elif answer.lower() == "n":
                raise ValueError("Directory is not deleted!")
    else:
        Path(sim_dir).mkdir(parents=True, exist_ok=True)
