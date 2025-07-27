#!/bin/bash
sinteractive --account=punim2612 --time=240 --ntasks=1 --cpus-per-task 8 -p gpu-a100-short,gpu-h100,gpu-a100 --gres=gpu:1 --mem-per-cpu=12G