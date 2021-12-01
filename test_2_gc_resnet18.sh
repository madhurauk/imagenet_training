#!/bin/bash
#SBATCH --job-name=in_rn18_resume
#SBATCH --output=GRADCAM_MAPS/resnet18/logs-%j.out
#SBATCH --error=GRADCAM_MAPS/resnet18/logs-%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
for i in {1..90}
do
# python test_2_gc_resnet18.py --resume "models/run4/" --evaluate --workers 4 /coc/scratch/mummettuguli3/data/imagenet
python test_2_gc_resnet18.py --resume "models/run4/model_state_epoch_${i}.pt" --evaluate --output_dir "GRADCAM_MAPS/resnet18/${SLURM_JOBID}" --workers 4 /coc/scratch/mummettuguli3/data/imagenet --dataset "imagenet" --class_list Samoyed robin chickadee indigo_bunting kite vulture mongoose
done

python create_gif.py --output_dir "GRADCAM_MAPS/resnet18/${SLURM_JOBID}/" --dataset "imagenet" --class_list Samoyed robin chickadee indigo_bunting kite vulture mongoose