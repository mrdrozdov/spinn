#!/bin/bash

# Generic job script for all experiments.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -t24:00:00

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 pytorch/intel/20170125 protobuf/intel/3.1.0

# Run.
python -m spinn.models.fat_classifier  --noshow_progress_bar --gpu 0 $SPINN_FLAGS


