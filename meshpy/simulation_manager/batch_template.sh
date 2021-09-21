#!/bin/sh -f
#
###############################
# Specify your SLURM directives
###############################
# User's Mail:
#SBATCH --mail-user=matthias.mayr@unibw.de
# When to send mail?:
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Job name:
#SBATCH --job-name "{job_name}"
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Define if the job should only be run on certain nodes.
#{is_feature}SBATCH --constraint={self.feature}
#
# Standard case: specify only number of cpus
# #SBATCH --ntasks=24
#
# If you want to specify a certain number of nodes
# and exactly 'ntasks-per-node' cpus on each node.
#SBATCH --nodes={self.n_nodes}
#SBATCH --ntasks-per-node={self.n_proc}
#
# For hybrid mpi: e.g. 2 mpi processes each with
# 4 openmp threads
# #SBATCH --ntasks=2
# #SBATCH --cpus-per-task=4
#
# Allocate full node and block for other jobs
#{is_exclusive}SBATCH --exclusive
#
# Walltime:
#SBATCH --time={self.wall_time}
###########################################

# Store calling directroy
CWD=$(pwd)

# Setup shell environment
echo $HOME
cd $HOME
source /etc/profile.d/modules.sh
source /home/opt/cluster_tools/core/load_baci_environment.sh

# Go back to calling directroy
cd  $CWD

########################
# GENERAL SPECIFICATIONS
########################
BUILD_DIR="$BACI_BUILD_DIR"
RUN_BACI="ON"
RUN_ENSIGHT_FILTER="OFF"

#####################
# INPUT SPECIFICATION
#####################
INPUT="$SIMULATIONS_DIR/{input_file_name}"

######################
# OUTPUT SPECIFICATION
######################
OUTPUT_PREFIX="{self.output_prefix}"
BACI_OUTPUT_DIR="$SIMULATIONS_DIR"

#######################
# RESTART SPECIFICATION
#######################
RESTART_FROM_STEP={self.restart_step}                 # <= specify your restart step
RESTART_FROM_DIR="{self.restart_dir}"
RESTART_FROM_PREFIX="{self.restart_from_prefix}" # <= specify the result prefix from which restart read

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# execute program
source /home/opt/cluster_tools/core/baci_job_core
trap 'early; stageout' 2 9 15 18
dochecks
stagein
runprogram
stageout
show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"
