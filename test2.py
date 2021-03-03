import torchvision.models as models
from gradcam import GradCAM
from gradcam.utils import visualize_cam
import torch
import PIL
from torchvision import transforms
from torchvision.utils import save_image
import pdb

args = {}
args.data = '/coc/scratch/mummettuguli3/data/imagenet'

# Data loading code
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

model = models.__dict__['resnet18'](pretrained=True)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('model_state.pt'))
model.eval()

#img_name = 'water-bird.jpeg'

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
save_image(heatmap, 'img1.png')
pdb.set_trace()

if __name__ == '__main__':
    main()