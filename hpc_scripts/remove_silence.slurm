#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o "run_logs/removesilence-%N.%j.out" #STDOUT
#SBATCH -e "run_logs/removesilence-%N.%j.err" #STDOUT

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 5:00:00

#SBATCH --mem-per-cpu=8G

#SBATCH --account=punim2612

# Specify your email address to be notified of progress.
#SBATCH --mail-user=james.quang@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load FFmpeg/5.0.1;
source ~/.venv/bin/activate
pip install -r ~/requirements.txt
python scripts/remove_silence.py
