#!/bin/bash
#SBATCH --job-name=hmn14
#SBATCH --output hmn14.log
#SBATCH --error hmn14.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=64G

cd ~/MLLM-Spurious
srun ~/micromamba/envs/spurious-test/bin/python ./pipeline/run_object_detector.py --dataset=hardimagenet-14 --model=owl

wait
