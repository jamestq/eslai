#!/bin/bash
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o "batch16.%j.out" #STDOUT
#SBATCH -e "batch16.%j.err" #STDERR

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 4:00:00

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
pip install -r ~/requirements.txt
python scripts/auplay.py process-audio accent_recognition/accent_train_noisy.tsv accent_recognition/audio_files_treated /tmp/punim2612/encoded_batch16
python scripts/auplay.py feature-extraction /tmp/punim2612/encoded_batch16 --model-name facebook/wav2vec2-large-xlsr-53 --output-dir /tmp/punim2612/feature_extracted_batch16
WANDB_API_KEY=5906bb4a857061a6ecf77b4b2b628e83975bf769 python scripts/auplay.py train-model /tmp/punim2612/feature_extracted_batch16 --output-dir model_output/batch16 --run-name wav2vec2-xlsr-batch16 --num-train-epochs 30 --learning-rate 1e-3
