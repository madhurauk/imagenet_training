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
python main_90_epochs_resnet50.py -a resnet50 --workers 40 --resume /srv/share3/mummettuguli3/code/thesis/models/run5/checkpoint.pth.tar /coc/scratch/mummettuguli3/data/imagenet