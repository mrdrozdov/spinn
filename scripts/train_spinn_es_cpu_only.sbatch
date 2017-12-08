#!/bin/bash

# Generic job script for all experiments.

#SBATCH --nodes=1 
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH -t48:00:00

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 pytorch/python2.7/0.3.0_4 protobuf/intel/3.1.0

# Default model.
MODEL="spinn.models.es_classifier"

# Optionally override default model.
if [ -n "$SPINNMODEL" ]; then
    MODEL=$SPINNMODEL
fi

# Run.
export IFS=';'
for SUB_FLAGS in $SPINN_FLAGS
do
	unset IFS
	python -m $MODEL --noshow_progress_bar $SUB_FLAGS
done

wait
