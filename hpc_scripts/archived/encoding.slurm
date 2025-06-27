#!/bin/bash
#SBATCH --ntasks=1 --time=0-15:00:00
module load GCCcore/11.3.0; module load Python/3.10.4; module load FFmpeg/5.0.1;
source ~/venvs/venv-3.10.4/bin/activate
pip install -r requirements.txt
python scripts/audio_encode.py