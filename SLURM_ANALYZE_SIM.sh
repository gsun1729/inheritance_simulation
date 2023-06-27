#!/bin/bash

#SBATCH --job-name=bulk_analysis
#SBATCH --time=48:00:00
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48

echo "====================================="    >> SLURM_ANALYZER_ERRORS.log

ml python/3.8.6 py-pip py-setuptools

python -m venv ./myenv
source ./myenv/bin/activate

pip install -r requirements.txt



git checkout main
echo "running analyze simulations bulk analysis" >> SLURM_ANALYZER_ERRORS.log
now=$(date)
echo "Current time : $now" >> SLURM_ANALYZER_ERRORS.log

echo "analyzing " >> SLURM_ANALYZER_ERRORS.log
python analyze_everything.py experiments/2>> SLURM_ANALYZER_ERRORS.log


now=$(date)
echo "Finish time : $now" >> SLURM_ANALYZER_ERRORS.log

