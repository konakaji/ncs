#!/bin/bash

#SBATCH -A m4461_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --image=ghcr.io/1tnguyen/cuda-quantum:mpich-231710
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --module=cuda-mpich
# srun -N 1 -n 4 python --version
srun -N 1 -n 1 shifter bash srun_exec.sh $1 --target nvidia-mgpu