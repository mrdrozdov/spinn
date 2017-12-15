#!/bin/bash

# Generic job script for all experiments.

#SBATCH --partition=mhealth,...
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -t24:00:00

#PRINCE PRINCE_GPU_COMPUTE_MODE=default

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

# Make sure we have access to HPC-managed libraries.
module load python3/intel/3.6.3 pytorch/python3.6/0.3.0_4

# Default model.
MODEL="spinn.models.supervised_classifier"

# Optionally override default model.
if [ -n "$SPINNMODEL" ]; then
    MODEL=$SPINNMODEL
fi

# Run.
export IFS=';'
for SUB_FLAGS in $SPINN_FLAGS
do
	unset IFS
	python3 -m $MODEL --noshow_progress_bar --gpu 0 $SUB_FLAGS &
done

wait 
