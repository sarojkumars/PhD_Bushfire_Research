#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00
#SBATCH --partition=physical

# Load required modules
module purge
module load foss/2019b 
module load python/3.7.4
module load netcdf4-python/1.4.3-python-3.7.4
module load scipy-bundle/2019.10-python-3.7.4

# Launch multiple process python code
echo "Searching for mentions"
time srun -n 2 python3 soil_temp_0p225m_hpc.py -i soil_temp_0p225m_hpc.csv -m
echo "Searching for topics"
time srun  -n 2 python3 soil_temp_0p675m_hpc.py -i soil_temp_0p675m_hpc.csv -m