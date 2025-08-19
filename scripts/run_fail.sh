#!/bin/bash
#PBS -P rp06
#PBS -q normal           
#PBS -l ncpus=64
#PBS -l mem=8GB
#PBS -l walltime=05:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

# Load required modules
# module load cuda/12.6.2
# module load python3/3.10.4

# Activate virtual environment
source /scratch/rp06/sl5952/Poly-CCA/.venv/bin/activate

# Change to project directory
cd ..

python3 ./run_failure_suite.py >> failure_suite.log 2>&1