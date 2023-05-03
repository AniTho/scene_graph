import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch import nn
import matplotlib.pyplot as plt
import networkx as nx

mean, std = [0.4194, 0.4612, 0.3479], [0.2038, 0.1898, 0.2043]
crop_size = 512
base_model = 'resnet50'
cat_to_idx =  {'Background':0, 'Building':1, 'Road':2, 'Water':3, 'Tree':4, 'Vehicle':5, 'Pool':6, 'Grass':7}
idx_to_cat = {v:k for k, v in cat_to_idx.items()}

def main(img):#, model_name = 'resnet50', model_path = 'saved_models/resnet50_dice_loss_19.pt'):
    # global base_model
    model_path = 'saved_models/resnet50_dice_loss_19.pt'
    aug = get_transforms()
    img_tensor = aug(image = img/255.)['image']
    model = get_model(model_path)
    logits = model(img_tensor[None])
    _, pred_mask = torch.max(logits[0], dim = 0)
    mask = pred_mask.cpu().detach().numpy()
    all_masks = get_object_masks(mask)
    objects_and_contours = get_instances(all_masks)
    graph, node_positions = generate_graph(objects_and_contours)
    # print(node_positions)
    save_results(img, mask, graph, node_positions)
    graph_over_image(img, graph, node_positions)
    final_img = cv2.imread('final_result.png')
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    scene_over = cv2.imread('graph_over.png')
    scene_over = cv2.cvtColor(scene_over, cv2.COLOR_BGR2RGB)
    return final_img, scene_over

def get_instances(all_masks):
    objects_and_contours = {}
    for idx, mask in enumerate(all_masks):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        i = 1
        for contour in contours:
            total_area = mask.shape[0]*mask.shape[1]
            contour_mask = np.zeros_like(mask)
            area = cv2.contourArea(contour)/(total_area)
            if area < 0.001:           # ignoring small contours     
                continue
            contour_mask = cv2.fillPoly(contour_mask, [contour], (255))
            object_name = f'{idx_to_cat[idx]}_{i}'
            i+=1
            objects_and_contours[object_name] = contour_mask
    return objects_and_contours

def generate_graph(objects_and_contours):
    graph = nx.Graph()
    node_positions = {}
    dilate_kernel = np.ones((5,5))
    for ref_img, ref_mask in objects_and_contours.items():
        moments = cv2.moments(ref_mask)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        node_positions[ref_img] = np.array([cX, cY])
        ref_mask = cv2.dilate(ref_mask, kernel=dilate_kernel, iterations=2)
        for it_img, it_mask in objects_and_contours.items():
            if it_img == ref_img:
                continue
            it_mask = cv2.dilate(it_mask, kernel=dilate_kernel, iterations=2)
            connected = np.any(np.logical_and(ref_mask, it_mask))
            if connected:
                edge = (it_img, ref_img)
                if not graph.has_edge(*edge):
                    graph.add_edge(*edge)

    return graph, node_positions

def graph_over_image(img, graph, node_positions):
    fig, ax = plt.subplots(figsize=(12, 8),tight_layout=True)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Scene Graph over Image',fontsize=20)
    pos = {}
    for key, value in node_positions.items():
        pos[key] = (value[0], img.shape[0]- value[1])
    nx.draw_networkx(graph, pos=node_positions, with_labels=True, font_size=10, node_size=150,font_weight='bold')
    fig.savefig('graph_over.png')
    plt.close(fig)

def save_results(img, mask, graph, node_positions):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,15), tight_layout=True)

    # Plot the image in the left subplot
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Image', fontdict={'fontsize': 30,'fontweight': 'bold'})
    ax2.imshow(mask, cmap = 'gray')
    ax2.axis('off')
    ax2.set_title('Mask', fontdict={'fontsize': 30,'fontweight': 'bold'})

    # Plot the graph in the right subplot
    pos = {}
    for key, value in node_positions.items():
        pos[key] = (value[0], img.shape[1]-value[1])
    nx.draw_networkx(graph, pos=pos, with_labels=True,ax=ax3,font_size=20,font_weight='bold',node_size=1000)
    ax3.set_title('Scene Graph', fontdict={'fontsize': 30,'fontweight': 'bold'})

    plt.savefig('final_result.png')
    plt.close(fig)

def get_object_masks(mask):
    all_masks = []
    kernel = np.ones((5,5))
    for i in range(10):
        if i == 1 or i == 3:                    # ignoring building_f and road_f
            continue
        mask_sub = (mask == i).astype(np.uint8)
        mask_sub = cv2.dilate(mask_sub, kernel=kernel, iterations = 3)
        all_masks.append(mask_sub)
    return all_masks

class SegmentationModel(nn.Module):
    def __init__(self, base_model):
        super(SegmentationModel,self).__init__()
        self.arc = smp.Unet(encoder_name = base_model, encoder_weights = 'imagenet', classes = 10)

    def forward(self, images):
        logits = self.arc(images)
        return logits

def get_transforms():
    augmentations = A.Compose([A.LongestMaxSize(max_size=crop_size, interpolation=1),
                                A.PadIfNeeded(min_height=crop_size, min_width=crop_size),
                                A.Normalize(mean = mean, std = std, max_pixel_value=1),
                                ToTensorV2(),
                                ])
    return augmentations

def get_model(base_path):
    model = SegmentationModel(base_model=base_model)
    model.load_state_dict(torch.load(base_path))
    return model

if __name__ == '__main__':
    # print(os.listdir('data/FloodNet-Supervised_v1.0/'))
    img = cv2.imread('data/FloodNet-Supervised_v1.0/test/test-org-img/6419.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    main(img)