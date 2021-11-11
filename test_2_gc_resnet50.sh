#!/bin/bash
#SBATCH --job-name=gc_in_rn50_resume
#SBATCH --output=GRADCAM_MAPS/resnet50/logs-%j.out
#SBATCH --error=GRADCAM_MAPS/resnet50/logs-%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=debug

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
#python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /coc/scratch/mummettuguli3/data/imagenet
# python test.py
for i in {1..90}
do
python test_2_gc_resnet50.py -a resnet50 --resume "models/run5/model_state_epoch_${i}.pt" --evaluate --output_dir "GRADCAM_MAPS/resnet50/${SLURM_JOBID}" --workers 4 /coc/scratch/mummettuguli3/data/imagenet
done