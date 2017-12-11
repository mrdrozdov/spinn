#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -t24:00:00


# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

module load python-2.7

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
	python -m $MODEL --noshow_progress_bar --gpu 0 $SUB_FLAGS &
done

wait 
