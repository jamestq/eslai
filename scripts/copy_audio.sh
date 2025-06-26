#!/bin/bash
input_dir=dataset
output_dir=accent_recognition/audio_files/
filelist=accent_recognition/"$1"

while IFS= read -r filepath; do
  fullpath="$input_dir"/"$filepath"
  if [[ -f "$fullpath" ]]; then
    cp "$fullpath" "$output_dir"
  else
    echo "Warning: $fullpath not found"
  fi
done < "$filelist"