#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o "run_logs/classifierxlsr-%N.%j.out" #STDOUT
#SBATCH -e "run_logs/classifierxlsr-%N.%j.err" #STDOUT

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 2:00:00

#SBATCH --mem-per-cpu=12G

# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-h100,gpu-a100-short
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
source ~/.venv/bin/activate
source scripts/remove_old_files.sh
pip install -r ~/requirements.txt
python scripts/auplay.py process-audio accent_recognition/accent_train.tsv accent_recognition/audio_files_treated accent_recognition/encoded
python scripts/auplay.py feature-extraction accent_recognition/encoded --model-name facebook/wav2vec2-large-xlsr-53 --output-dir accent_recognition/feature_extracted
WANDB_API_KEY=5906bb4a857061a6ecf77b4b2b628e83975bf769 python scripts/auplay.py train-model accent_recognition/feature_extracted --model-name facebook/wav2vec2-large-xlsr-53 --output-dir model-output --train-batch-name wav2vec2-xlsr-custom-sched --num-train-epochs 30
