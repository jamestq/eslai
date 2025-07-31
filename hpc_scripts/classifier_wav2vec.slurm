#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o "run_logs/classifierwav2vec2-%N.%j.out" #STDOUT
#SBATCH -e "run_logs/classifierwav2vec2.%j.err" #STDOUT

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
source ~/.venv/bin/activate
pip install -r ~/requirements.txt
python scripts/auplay.py process-audio accent_recognition/accent_train.tsv accent_recognition/audio_files_treated /tmp/punim2612/encoded_"$1"
python scripts/auplay.py feature-extraction /tmp/punim2612/encoded_"$1" --model-name facebook/wav2vec2-base --output-dir /tmp/punim2612/feature_extracted_"$1"
WANDB_API_KEY=5906bb4a857061a6ecf77b4b2b628e83975bf769 python scripts/auplay.py train-model /tmp/punim2612/feature_extracted_"$1" --output-dir model_output_"$1" --run-name wav2vec2-base-"$1" --num-train-epochs $2 --learning-rate $3
