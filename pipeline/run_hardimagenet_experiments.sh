#!/bin/bash
#SBATCH --job-name=h14ld
#SBATCH --output h14ld.log
#SBATCH --error h14ld.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=64G

cd ~/MLLM-Spurious
srun ~/micromamba/envs/spurious-test/bin/python ./pipeline/run_experiments.py --dataset=hardimagenet-14 --mllm=llava --img_type=dropped

wait
