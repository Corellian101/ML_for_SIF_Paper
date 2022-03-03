#!/bin/bash
#SBATCH --time=222:00:00 # walltime, abbreviated by -t
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=legojonas@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks=12
# additional information for allocated clusters
#SBATCH --account=hochhalter-np     # account - abbreviated by -A
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p

export WORKDIR=$PWD
export SCRDIR=/scratch/kingspeak/serial/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR

source /uufs/chpc.utah.edu/common/home/u6019587/.venv/bingovenv_v2/bin/activate
export PYTHONPATH=/uufs/chpc.utah.edu/common/home/u1164746/bingo


export MV2_ENABLE_AFFINITY=0

mpiexec -n $SLURM_NTASKS python -m mpi4py bingo_evolution.py

cp -r $SCRDIR/* $WORKDIR
