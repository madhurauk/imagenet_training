#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
python main_90_epochs_resnet50.py -a resnet50 --dist-url 'tcp://127.0.0.1:8877' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 40 /coc/scratch/mummettuguli3/data/imagenet