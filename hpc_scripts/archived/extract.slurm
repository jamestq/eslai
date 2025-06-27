#!/bin/bash
#SBATCH --ntasks=1
tmpdir=/tmp/punim2612/CommonVoice
outdir=/data/gpfs/projects/punim2612/CommonVoice/dataset
filelist=filelist1.txt

echo "Creating tempoprary directories"
mkdir -p $tmpdir
mkdir -p  $outdir
tar -xvzf CommonVoice.tar.gz -C $tmpdir
while read -r filepath; do    
    # Copy the file to the destination directory
    fullpath="$tmpdir/$filepath"
    # Check if the file exists before copying
    if [[ -f "$fullpath" ]]; then
        cp "$fullpath" "$outdir"
    else
        echo "Warning: $fullpath not found"
done < "$filelist"
