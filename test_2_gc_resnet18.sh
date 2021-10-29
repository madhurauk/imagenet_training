#!/bin/bash
#SBATCH --job-name=in_rn18_resume
#SBATCH --output=GRADCAM_MAPS/resnet18/logs-%j.out
#SBATCH --error=GRADCAM_MAPS/resnet18/logs-%j.err
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --partition=debug

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
# for i in {1..90}
# do
python test_2_gc_resnet18.py --resume "models/run4/" --evaluate --workers 4 /coc/scratch/mummettuguli3/data/imagenet
# done