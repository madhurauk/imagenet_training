#!/bin/bash
#SBATCH --job-name=not_mp_imagenet_training_resnet18
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
python main_90_epochs.py -a resnet18 --workers 40 --resume /srv/share3/mummettuguli3/code/thesis/models/run4/checkpoint.pth.tar /coc/scratch/mummettuguli3/data/imagenet