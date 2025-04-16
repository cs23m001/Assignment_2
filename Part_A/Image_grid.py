import os
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import dataset, class_labels
from model import Model


transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
test_dataset = dataset.iNaturalist(img_path =  os.path.join(os.getcwd(),'utils','test_data.csv'), transforms = transform)
test_loader = DataLoader(test_dataset, batch_size = 30, shuffle = True)

# Fetching the batch of 30 images and label
for batch_idx, (images, labels) in enumerate(test_loader):
    if batch_idx == 0:
        break

# Reshaping C*H*W to  H*W*C 
images_actual = images.permute(0, 2, 3, 1)

# Applying normalization transform
norm_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sending to gpu
images_norm = norm_transform(images)
images_norm = images_norm.to(device=device)

# Defineing the model structure
model = Model(filters = [32, 64, 64, 128, 256], kernel_size = [3, 3, 3, 3, 3], dense_size = 128, batch_norm = True, dropout = 0.1)
model.to(device)
# loading the pre-trained weights
# you have to set the path accordingly
model.load_state_dict(torch.load('part_a_best_model.pth'))

# prediction
prediction = model(images_norm)
_, prediction = torch.max(prediction, dim=1)
prediction = prediction.cpu()
print(labels)
print(prediction)
print((prediction == labels).sum())

# Ploting Vertical
fig, axs = plt.subplots(10, 3, figsize=(8,10))

labels = labels.detach().cpu().numpy()
prediction = prediction.detach().cpu().numpy()

k = 0
for i in range(10):
    for j in range(3):
        ax = axs[i, j]
        ax.imshow(images_actual[i * 3 + j])
        ax.axis('off')  # Turn off axes
        color = 'green' if labels[k] == prediction[k] else 'red'
        ax.text(0.5, -0.2, f'Actual: {class_labels.Class_Label[labels[k]]}', horizontalalignment='center', transform=ax.transAxes, fontsize=6, color=color)
        ax.text(0.5, -0.4, f'Predicted: {class_labels.Class_Label[prediction[k]]}', horizontalalignment='center', transform=ax.transAxes, fontsize=6, color=color)
        k+=1


plt.tight_layout()
# plt.savefig('/home/apu/Desktop/fig/image_grid_with_labels_vertical.png', dpi=300)  
plt.show()


# Ploting Horizontal 3 *10 grid

# fig, axs = plt.subplots(3, 10, figsize=(12, 8))

# k = 0
# # Populate the grid with images
# for i in range(3):
#     for j in range(10):
#         ax = axs[i, j]
#         ax.imshow(images_actual[i * 10 + j]) 
#         ax.axis('off')  # Turn off axes
#         color = 'green' if labels[k] == prediction[k] else 'red'
#         ax.text(0.5, -0.2, f'Actual: {class_labels.Class_Label[labels[k]]}', horizontalalignment='center', transform=ax.transAxes, fontsize=6, color=color)
#         ax.text(0.5, -0.4, f'Predicted: {class_labels.Class_Label[prediction[k]]}', horizontalalignment='center', transform=ax.transAxes, fontsize=6, color=color)
#         k+=1

# plt.tight_layout()
# # plt.savefig('/home/apu/Desktop/fig/image_grid_with_labels_vertical.pngimage_grid_with_labels_horizontal.png', dpi=300)  
# plt.show()