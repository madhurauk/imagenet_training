import argparse
import pdb
import os

import sys
sys.path.append('/srv/share3/mummettuguli3/code/')
from utils.grad_cam_caller import ImageHelper

parser = argparse.ArgumentParser(description='generate pdf of gradcam maps')
parser.add_argument('--output_dir', default='', type=str, metavar='PATH',
                    help='path until output dir')
parser.add_argument('--job_id_list', default='', type=str, metavar='PATH',
                    help='list of job ids for which pdf is to be generated')

def main():
    args = parser.parse_args()
    img_helper = ImageHelper()
    class_types = ['ground_truth', 'predicted_class']
    for job_id in args.job_id_list.split(','):
        sub_path = os.path.join(args.output_dir, job_id)
        sub_path_folders = next(os.walk(sub_path))[1]
        for folder in sub_path_folders:
            path = os.path.join(sub_path, folder)
            img_helper.view_gcam_generate_pdf(path, class_types[0])
            img_helper.view_gcam_generate_pdf(path, class_types[1])


if __name__ == '__main__':
    main()