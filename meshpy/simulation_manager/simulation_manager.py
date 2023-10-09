# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2023
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
A class that creates scripts to run multiple simulations.
"""


# Import python modules.
import os
import subprocess
import time
import builtins


def is_true_batch(value):
    """Return a batch comment if value is False."""
    if value:
        return ""
    else:
        return " #"


def wait_for_jobs_to_finish(job_ids, *, check_interval=10, status=False):
    """
    Wait until all jobs in job_ids are finished.

    Args
    ----
    job_ids: list
        List with all job IDs
    check_interval: int, float
        Interval in seconds where the job status is read.
    status: bool
        If a status of the simulation state should be written to the
        console.
    """

    def print(*args, **kwargs):
        """Overwrite the print function for this method."""
        if status:
            builtins.print(*args, **kwargs)

    # Wait for the jobs to finish.
    while True:

        # Get the currently running jobs.
        job_out = (
            subprocess.check_output(["squeue"], stderr=subprocess.STDOUT)
            .decode("UTF-8")
            .strip()
            .split("\n")
        )
        job_out = job_out[1:]
        active_jobs = []
        for job in job_out:
            active_jobs.append(int(job.strip().split(" ")[0]))

        # Check if the job IDs from the simulation are still active.
        my_active_jobs = []
        for job_id in job_ids:
            if job_id in active_jobs:
                my_active_jobs.append(job_id)

        if len(my_active_jobs) == 0:
            print("All jobs finished")
            break
        else:
            print(
                "{}/{} jobs finished".format(
                    len(job_ids) - len(my_active_jobs), len(job_ids)
                )
            )

        time.sleep(check_interval)


class Simulation:
    """
    Represents a single simulation.
    """

    def __init__(
        self,
        input_file_path,
        *,
        n_proc=None,
        n_nodes=None,
        n_proc_per_node=None,
        exclusive=False,
        output_prefix="xxx",
        wall_time="08:00:00",
        restart_step=0,
        restart_dir="",
        restart_from_prefix="",
        job_name=None,
        feature=None
    ):
        """
        Initialize the simulation object. The input file will be written to
        the specified path.

        Args
        ----
        input_file_path:
            Path on the disk where the input file is stored.
        n_proc: int
            Number of processors to be used for this simulation.
        n_nodes: int
            Number of nodes that the job should be run on.
        exclusive: bool
            If the job should be run exclusively at a node, i.e. no other jobs
            can be run at the same node during the job execution.
        output_prefix: str
            Name of the baci simulation.
        wall_time: str
            Maximum time for computation.
        job_name: str
            Name of this simulation.
        restart_step: int
            Step from where the restart should be started.
        restart_dir: str
            Directory where the previous simulation was performed.
        restart_from_prefix: str
            Name of the previous simulation.
        feature: str
            If the job can only be run on certain nodes, configured via a slurm
            feature.
        """

        # Class variables
        self.file_path = input_file_path
        self.n_proc = n_proc
        self.n_nodes = n_nodes
        self.n_proc_per_node = n_proc_per_node
        self.exclusive = exclusive
        self.output_prefix = output_prefix
        self.wall_time = wall_time
        self.job_name = job_name
        self.feature = feature

        # Consistency check
        if self.n_proc is not None and (
            self.n_nodes is not None or self.n_proc_per_node is not None
        ):
            raise ValueError(
                "The options n_proc and (n_nodes + n_proc_per_node) are mutually exclusive"
            )
        elif (self.n_nodes is None and self.n_proc_per_node is not None) or (
            self.n_nodes is not None and self.n_proc_per_node is None
        ):
            raise ValueError("Both options n_nodes and n_proc_per_node are required")

        # Set default value for n_proc
        if (
            self.n_proc is None
            and self.n_nodes is None
            and self.n_proc_per_node is None
        ):
            self.n_proc = 1

        # Restart options
        self.restart_step = restart_step
        self.restart_dir = restart_dir
        self.restart_from_prefix = restart_from_prefix

    def create_run_script(self, base_dir, *, quiet=True):
        """
        Create the command lines to run the simulation.

        Args
        ----
        base_dir: str
            Directory where the simulation should be performed.
        quiet: bool
            If the baci output should be redirected to the console.
        """

        run_script = "cd $SIMULATIONS_BASE_DIR\n"
        run_script += "cd {}\n".format(
            os.path.relpath(os.path.dirname(self.file_path), base_dir)
        )
        if self.restart_step == 0:
            baci_command = (
                "mpirun -np {self.n_proc} $BACI_WORK_RELEASE "
                + "{input_file} {self.output_prefix}"
            )
        else:
            baci_command = (
                "mpirun -np {self.n_proc} $BACI_WORK_RELEASE "
                + "{input_file} {self.output_prefix} "
                + "restart={self.restart_step} "
                + "restartfrom="
                + "{self.restart_dir}/{self.restart_from_prefix}"
            )
        if quiet:
            run_string = baci_command + " > {self.output_prefix}.log\n"
        else:
            run_string = baci_command + " | tee {self.output_prefix}.log\n"
        run_script += run_string.format(
            self=self,
            input_file=os.path.basename(self.file_path),
            baci_command=baci_command,
        )
        return run_script

    def create_batch_file(self, base_dir, batch_name):
        """
        Create the batch file to submit this simulation in slurm.

        Args
        ----
        base_dir: str
            Base directory of the simulation
        bath_name: str
            Name of the created batch file.
        """

        # Load the batch template.
        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "batch_template.sh"
            ),
            "r",
        ) as file:
            batch_string = file.read()

        # Get the relative path from this simulation to the base directory.
        rel_path = os.path.relpath(os.path.dirname(self.file_path), base_dir)
        if self.job_name is None:
            simulation_name = os.path.basename(self.file_path)
            job_name = rel_path + "/" + simulation_name
        else:
            job_name = self.job_name
        # Slurm does not accept job names with "/" so we replace them with a dash here
        job_name = job_name.replace("/", "-")

        batch_string = batch_string.format(
            self=self,
            job_name=job_name,
            input_file_name=os.path.basename(self.file_path),
            is_exclusive=is_true_batch(self.exclusive),
            is_feature=is_true_batch(self.feature is not None),
            is_node=is_true_batch(self.n_nodes is not None),
            is_not_node=is_true_batch(self.n_nodes is None),
        )
        batch_script_path = os.path.join(os.path.dirname(self.file_path), batch_name)
        with open(batch_script_path, "w") as file:
            file.write(batch_string)

        run_script = "export SIMULATIONS_DIR=$SIMULATIONS_BASE_DIR/{}\n".format(
            rel_path
        )
        run_script += "cd $SIMULATIONS_DIR\n"
        run_script += 'sbatch "$SIMULATIONS_DIR/{}"\n'.format(batch_name)
        return run_script


class SimulationManager:
    """
    This class manages multiple simulations and can create scripts to execute
    them all.
    """

    def __init__(self, path):
        """
        Initialize the object.

        Args
        ----
        path: str
            Path where the simulation manager will create its run scripts.
        """

        # Class variables.
        self.path = path
        self.simulations = []

    def add(self, other):
        """
        Add objects to the manager. Currently only Simulations and lists of
        Simulations can be added. All added simulations have to be in a sub
        directory from the manager path.
        """

        if isinstance(other, Simulation):
            if os.path.realpath(other.file_path).startswith(
                os.path.realpath(self.path)
            ):
                self.simulations.append(other)
            else:
                raise ValueError(
                    "The path of the simulation is not a sub path of this manager"
                )
        elif isinstance(other, list):
            for item in other:
                self.add(item)
        else:
            raise TypeError("Expected Simulation or list of Simulations")

    def create_run_scripts(
        self,
        *,
        baci_build_dir=None,
        status=False,
        status_file=False,
        script_name="run.sh",
        **kwargs
    ):
        """
        Create the script to run all simulations contained in this manager.

        Args
        ----
        baci_build_dir: str
            Build directory for baci.
        status: bool
            If the status sould be output during the simulation. This will
            set the quiet option for the individual simulations.
        status_file: bool
            If a status file should be created with the current state of the
            simulations.
        script_name: str
            Name of the created run script.
        """

        run_script = ""
        if baci_build_dir is None:
            run_script += "BACI_WORK_RELEASE=<set baci path here>\n"
        else:
            run_script += "BACI_WORK_RELEASE={}/baci-release\n".format(baci_build_dir)
        run_script += "SIMULATIONS_BASE_DIR={}\n\n".format(self.path)

        if status_file:
            run_script += "rm -f run_status.log\n"

        for simulation in self.simulations:
            if status_file:
                run_script += "cd {}\n".format(self.path)
                run_script += 'echo "{}" >> run_status.log\n'.format(
                    simulation.file_path
                )
            if status:
                run_script += 'echo "{}"\n'.format(simulation.file_path)
            run_script += simulation.create_run_script(self.path, **kwargs)
            if status_file:
                run_script += "cd {}\n".format(self.path)
                run_script += 'echo "done" >> run_status.log\n'
            if status:
                run_script += 'echo "done"\n'

        run_script_path = os.path.join(self.path, script_name)
        with open(run_script_path, "w") as file:
            file.write(run_script)
        os.chmod(run_script_path, 0o764)
        return run_script_path

    def create_batch_scripts(
        self, *, baci_build_dir=None, script_name="run.sh", batch_name="batch.sh"
    ):
        """
        Create cluster batch scripts to run all simulations contained in this
        manager.

        Args
        ----
        baci_build_dir: str
            Path to the baci build directory.
        script_name: str:
            Name of script to submit batch files.
        bath_name: str
            Name of the created batch file.
        """

        run_script = ""
        if baci_build_dir is None:
            run_script += "export BACI_BUILD_DIR="
            run_script += "<set baci path build directory here>\n"
        else:
            run_script += "export BACI_BUILD_DIR={}\n".format(baci_build_dir)
        run_script += "export SIMULATIONS_BASE_DIR=$(readlink -f $(dirname $0))\n\n"

        for simulation in self.simulations:
            run_script += simulation.create_batch_file(self.path, batch_name)

        run_script_path = os.path.join(self.path, script_name)
        with open(run_script_path, "w") as file:
            file.write(run_script)
        os.chmod(run_script_path, 0o764)
        return run_script_path

    def submit_batch_files(self, **kwargs):
        """
        Execute all simulations in the manager and return the resulting job
        IDs.
        """

        # Create the batch files.
        run_script = self.create_batch_scripts(**kwargs)

        # Execute the created script if sbatch is available on the system.
        run_out = (
            subprocess.check_output(run_script, shell=True, stderr=subprocess.STDOUT)
            .decode("UTF-8")
            .strip()
        )

        # Get the job IDs.
        jobs = run_out.split("\n")
        job_ids = [int(job.strip().split(" ")[-1]) for job in jobs]
        return job_ids

    def submit_batch_files_and_wait_for_finish(self, *, baci_build_dir=None, **kwargs):
        """
        Execute all simulations the run_script and wait for the jobs to finish.

        Args
        ----
        baci_build_dir: str
            Path the baci build directory.
        check_interval: int, float
            Interval in seconds where the job status is read.
        status: bool
            If a status of the simulation state should be written to the
            console.
        """

        # Submit the jobs and wait for them to finish.
        wait_for_jobs_to_finish(
            self.submit_batch_files(baci_build_dir=baci_build_dir), **kwargs
        )

    def run_simulations_and_wait_for_finish(
        self, *, slurm=False, status=True, **kwargs
    ):
        """
        Execute all simulations in this manager.

        Args
        ----
        slurm: bool
            If the simulations should be run directly or over slurm.
        status: bool
            If status updates about the state of the simulation should be
            given.
        """

        if slurm:
            self.submit_batch_files_and_wait_for_finish(status=status, **kwargs)
        else:
            run_script_path = self.create_run_scripts(status=status, **kwargs)
            subprocess.run(run_script_path, shell=True)
