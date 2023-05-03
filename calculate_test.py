import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss
import gc
import wandb
import random

class SatelliteDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, images, transforms = None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
         
    def __getitem__(self, idx):
        current_img_path = self.images[idx]
        current_label_path = self.label_mask_path(current_img_path)
        img = np.asarray(Image.open(current_img_path)).copy() / 255.
        mask = np.asarray(Image.open(current_label_path)).copy()
        if self.transforms:
            transformed = self.transforms(image = img, mask = mask)
            img = transformed['image']
            mask = transformed['mask']
        return img, mask

    def label_mask_path(self, image_path):
        return self.masks_dir/f'{image_path.stem}_lab.png'

class SegmentationModel(nn.Module):
    def __init__(self, base_model):
        super(SegmentationModel,self).__init__()
        self.arc = smp.Unet(encoder_name = base_model, encoder_weights = 'imagenet', classes = 10)

    def forward(self, images):
        logits = self.arc(images)
        return logits
    
def label_mask_path(mask_dir, image_path):
    return mask_dir/f'{image_path.stem}_lab.png'

#For deterministic behavious
seed = 42
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)

batch_size = 16
img_resize = 560
crop_size = 512
mean, std = [0.4194, 0.4612, 0.3479], [0.2038, 0.1898, 0.2043]
base_data_dir = pathlib.Path('data/FloodNet-Supervised_v1.0/')
test_dir = base_data_dir/'test'
test_img_dir, test_mask_dir = test_dir/'test-org-img', test_dir/'test-label-img'
test_images = list(test_img_dir.glob('*.jpg'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting first experiment
wandb.init(project='scene_segmentation', name=f"test_data_result")
models_best = {'resnet34_focal_loss': 14, 'resnet34_dice_loss':19, 'resnet50_focal_loss':18, 'resnet50_dice_loss':19}
test_augmentation = A.Compose([A.LongestMaxSize(max_size=crop_size, interpolation=1),
                                A.PadIfNeeded(min_height=crop_size, min_width=crop_size),
                                A.Normalize(mean = mean, std = std, max_pixel_value=1),
                                ToTensorV2(),
                                ])
test_dataset = SatelliteDataset(test_img_dir, test_mask_dir, test_images, test_augmentation)
testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

with torch.no_grad():
    for base_model, iter in models_best.items():
        print(f'****** {base_model} ******')
        model_name = base_model[:base_model.index('_')]
        model = SegmentationModel(model_name)
        model = model.to(device)
        model.load_state_dict(torch.load(f'saved_models/{base_model}_{iter}.pt'))
        model.eval()
        total_iou = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        for imgs, masks in tqdm(testloader, leave=False, total=len(testloader)):
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)
            out = torch.max(out, dim = 1)[1]
            tp, fp, fn, tn = smp.metrics.get_stats(out, masks, mode = 'multiclass', num_classes=10)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro-imagewise')
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction='macro-imagewise')
            f1_score = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")
            total_iou+=iou_score
            total_recall+=recall
            total_f1 += f1_score
        
        total_iou = total_iou/len(testloader)
        total_recall = total_recall/len(testloader)
        total_f1 = total_f1/len(testloader)
        wandb.log({f'{base_model}_iou': total_iou, f'{base_model}_recall':total_recall, f'{base_model}_f1': total_f1})
        print(f'{base_model}_iou: {total_iou}, {base_model}_recall:{total_recall}, {base_model}_f1: {total_f1}')
    wandb.finish()
