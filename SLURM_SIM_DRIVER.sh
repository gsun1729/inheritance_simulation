#!/bin/bash

#SBATCH --job-name=single_component_sensitivity_analysis
#SBATCH --time=48:00:00
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48

ml python/3.8.6 py-pip py-setuptools

python -m venv ./myenv
source ./myenv/bin/activate

pip install -r requirements.txt



git checkout main
echo "running single_component_sensitivity_analysis"
now=$(date +"%T")
echo "Current time : $now"

python -O single_component_sensitivity_analysis.py experiments/ 2>> SLURM_SIM_DRIVER_ERRORS.log


now=$(date +"%T")
echo "Finish time : $now"

