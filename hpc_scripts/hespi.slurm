#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 2:00:00

#SBATCH --mem-per-cpu=12G

# SBATCH -p cascade
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim2612

# Specify your email address to be notified of progress.
#SBATCH --mail-user=james.quang@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load FFmpeg/5.0.1;
source ~/venvs/venv-3.10.4/bin/activate
pip install -r requirements.txt
hespi-tools sheet-component-location

yolo detect train data=glabels/yolo_config.yml  model=/home/quangu/.cache/hespi/sheet-component-fdbb302d891ca1b2d1045725ba6b562a.pt  epochs=50 imgsz=640 batch=4
