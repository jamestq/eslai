#!/bin/bash
tmpdir=/tmp/punim2612/CommonVoice
outdir=/data/gpfs/projects/punim2612/CommonVoice/dataset
filelist=filelist1.txt

echo "Creating temporary directories"
mkdir -p $tmpdir
mkdir -p  $outdir

while IFS= read -r filepath; do
    # Copy the file to the destination directory
    fullpath="$tmpdir/$filepath"
    # Check if the file exists before copying
    if [[ -f "$fullpath" ]]; then
        cp "$fullpath" "$outdir"
    else
        echo "Warning: $fullpath not found"
    fi
done < "$1"
