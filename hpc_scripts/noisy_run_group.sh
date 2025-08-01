#!/bin/bash
sbatch hpc_scripts/noisy_run/xlsr_lr_batch8.slurm &
sbatch hpc_scripts/noisy_run/xlsr_lr_batch16.slurm &
sbatch hpc_scripts/noisy_run/xlsr_lr_batch32.slurm &
sbatch hpc_scripts/noisy_run/xlsr_lr_batch64.slurm