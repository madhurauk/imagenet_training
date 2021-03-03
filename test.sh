#!/bin/bash
#SBATCH --job-name=imagenet_gradcam_resnet18
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=debug

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
#python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /coc/scratch/mummettuguli3/data/imagenet
python test.py