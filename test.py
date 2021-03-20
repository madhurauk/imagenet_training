import torchvision.models as models
from gradcam import GradCAM
from gradcam.utils import visualize_cam
import torch
import PIL
from torchvision import transforms
from torchvision.utils import save_image
import pdb

model = models.__dict__['resnet18'](pretrained=True)
model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load('model_state.pt'))
model.load_state_dict(torch.load('models/model_state_epoch_1.pt'))
# pdb.set_trace()
model.eval()

# # img_name = 'water-bird.jpeg'
img_name = 'n15075141_29199.JPEG'

pil_img = PIL.Image.open(img_name)
torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(pil_img).to('cuda')
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

print(model)
config = dict(model_type='resnet', arch=model.module, layer_name='layer4')
gc = GradCAM.from_config(**config)
mask, _ = gc(normed_torch_img)
heatmap, result = visualize_cam(mask, torch_img)
# img1=transforms.ToPILImage()(heatmap)
save_image(heatmap, 'img2.png')
# pdb.set_trace()
