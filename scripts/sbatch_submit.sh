#!/bin/bash

# Sets up a series of 10 jobs to run one after the other to get around time limits.


bat=../scripts/train_spinn.sbatch

jobID=
for((i=0; i<2; i++)); do
    if [ "$jobID" == "" ]; then
        jobID=$(sbatch $bat | awk '{print $NF}')
    else
        jobID=$(sbatch --dependency=afterany:$jobID $bat | awk '{print $NF}')
    fi
done
