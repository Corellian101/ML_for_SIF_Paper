#!/bin/bash
#SBATCH --time=72:00:00 # walltime, abbreviated by -t
#SBATCH --mail-type=END,FAIL
####SBATCH --mail-user=legojonas@gmail.com
#SBATCH --nodes=1
#SBATH --mem 16000
#SBATCH --ntasks=8
# additional information for allocated clusters
#SBATCH --account=hochhalter-np     # account - abbreviated by -A
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p
##SBATCH --account=hochhalter
##SBATCH --partition=kingspeak

module load abaqus/2019

unset SLURM_GTIDS

abaqus cae noGUI=test_noCrop.py
