#!/bin/bash
input_dir=/data/gpfs/projects/punim2612/CommonVoice/dataset
output_dir=/data/gpfs/projects/punim2612/CommonVoice/accent_recognition/audio_files
filelist=/data/gpfs/projects/punim2612/CommonVoice/accent_recognition/"$1"

while IFS= read -r filepath; do
  fullpath="$input_dir/$filepath"
  # outpath="$output_dir/$filepath"
  if [[ -f "$fullpath" ]]; then
    cp "$fullpath" "$output_dir"
  else
    echo "Warning: $fullpath not found"
  fi
done < "$filelist"