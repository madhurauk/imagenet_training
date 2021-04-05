#!/bin/bash
#SBATCH --job-name=imagenet_training_resnet18
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
python main_90_epochs.py -a resnet18 --dist-url 'tcp://127.0.0.1:8899' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 40 /coc/scratch/mummettuguli3/data/imagenet