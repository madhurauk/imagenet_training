#!/bin/bash
#SBATCH --job-name=generate_pdf
#SBATCH --output=GRADCAM_MAPS/resnet18/generate_pdf_logs/logs-%j.out
#SBATCH --error=GRADCAM_MAPS/resnet18/generate_pdf_logs/logs-%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
python generate_pdf.py --output_dir "GRADCAM_MAPS/resnet18/" --job_id_list '155590','160284','160288'