#!/bin/bash

#SBATCH --job-name=analyze_inheritance
#SBATCH --time=48:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48

ml python/3.8.6 py-pip py-setuptools

python -m venv ./myenv
source ./myenv/bin/activate

pip install -r requirements.txt



git checkout main
now=$(date +"%T")
echo "Running analyze_inheritance"
echo "Current time : $now"

python analyze_inheritance.py experiments/ 2>> SLURM_ANALYZER_ERRORS.log

now=$(date +"%T")
echo "Finish time : $now"

