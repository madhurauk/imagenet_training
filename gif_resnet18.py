import os
import imageio
path = "GRADCAM_MAPS/resnet18/"
files = []
images = []
j = 1
# for i in os.listdir(path):
#    filepath = os.path.join(path,i)
#    if os.path.isfile(filepath) and '-{}-gradcam-'.format(j) in i:                                                                           
#       images.append(imageio.imread(filepath))
#       if j == 90:
#          break
#       j+=1

for i in os.listdir(path):
   filepath = os.path.join(path,i)
   if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
      files.append(filepath)

files.sort()


