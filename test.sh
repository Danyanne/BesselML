#!/bin/bash
#SBATCH --job-name=Optuna_run_22_08_14_25   # name + DD_MM_HH_MM (minutes in the end)
#SBATCH --time=03:00:00        # set a limit on the total run time
#SBATCH --partition=cpu        # specify partition name (cpu/gpu)
#SBATCH --ntasks=1             # specify number of (MPI) processes
#SBATCH --cpus-per-task=64     # specify amount of cpu cores per task
#SBATCH --mem-per-cpu=2G        # specify amount of RAM memory

echo "sbatch-INFO: start of job"
echo "sbatch-INFO: nodes: ${SLURM_JOB_NODELIST}"
echo "sbatch-INFO: system: ${SLURM_CLUSTER_NAME}"
cd /home/ctvrta/BesselML/source/
source ../venv/bin/activate

srun python3.12 optuna_hyperparams_search.py 

echo "sbatch-INFO: we're done"
date