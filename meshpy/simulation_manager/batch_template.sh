#!/bin/bash
##########################################
#                                        #
#  Specify your SLURM directives         #
#                                        #
##########################################
# User's Mail:
#SBATCH --mail-user=<your_email@provider.com>
# When to send mail?:
#SBATCH --mail-type=NONE
#
# Job name:
#SBATCH --job-name={job_name}
#
# Output file:
#SBATCH --output=slurm-%j-%x.out
#
# Error file:
#SBATCH --error=slurm-%j-%x.err
#
# Standard case: specify only number of cpus
# #SBATCH --ntasks=8
#
# Walltime: (days-hours:minutes:seconds)
#SBATCH --time={self.wall_time}
#
##########################################
#                                        #
#  Advanced SLURM settings	          #
#                                        #
##########################################
#
# If you want to specify a certain number of nodes:
#SBATCH --nodes={self.n_nodes}
#
# and exactly 'ntasks-per-node' cpus on each node:
#SBATCH --ntasks-per-node={self.n_proc}
#
# Allocate full node and block for other jobs:
#{is_exclusive}SBATCH --exclusive
#
# Request specific hardware features:
#{is_feature}SBATCH --constraint={self.feature}
#
###########################################

# Setup shell environment and start from home dir
echo $HOME
cd $HOME
source /etc/profile.d/modules.sh
source /home/opt/cluster_tools/core/load_baci_environment.sh

module list

##########################################
#                                        #
#  Specify the paths                     #
#                                        #
##########################################

RUN_BACI="ON"
BACI_BUILD_DIR="$BACI_BUILD_DIR"
EXE=$BACI_BUILD_DIR/baci-release

INPUT="$SIMULATIONS_DIR/{input_file_name}"
BACI_OUTPUT_DIR="$SIMULATIONS_DIR"
OUTPUT_PREFIX="{self.output_prefix}"

# Start everything from the simulation directory
cd $SIMULATIONS_DIR

##########################################
#                                        #
#  Postprocessing                        #
#                                        #
##########################################

RUN_ENSIGHT_FILTER="OFF"
ENSIGHT_OUTPUT_DIR=""
ENSIGHT_OPTIONS=""


##########################################
#                                        #
#  RESTART SPECIFICATION                 #
#                                        #
##########################################

RESTART_FROM_STEP={self.restart_step}            # specify the restart step here and in .datfile
RESTART_FROM_DIR="{self.restart_dir}"            # same as output
RESTART_FROM_PREFIX="{self.restart_from_prefix}" # prefix typically xxx

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################

# execute program
source /home/opt/cluster_tools/core/charon_job_core
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
StageOut
# show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"
