#!/bin/bash
#SBATCH --job-name=Test   # specify job name
#SBATCH --time=02:10:33        # set a limit on the total run time
#SBATCH --partition=cpu        # specify partition name (cpu/gpu)
#SBATCH --ntasks=1             # specify number of (MPI) processes
#SBATCH --cpus-per-task=32     # specify amount of cpu cores per task

echo "sbatch-INFO: start of job"
echo "sbatch-INFO: nodes: ${SLURM_JOB_NODELIST}"
echo "sbatch-INFO: system: ${SLURM_CLUSTER_NAME}"
cd /home/ctvrta/BesselML/source/
source /venv/bin/activate

python3.12 

echo "sbatch-INFO: we're done"
date