import os
import torch
import wandb
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
test_loader = DataLoader(test_dataset, batch_size =30, shuffle = True)

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
model = Model(filters = [32, 64, 64, 128, 128], kernel_size = [3, 3, 3, 5, 5], dense_size = 512, batch_norm = True, dropout = 0.45)
model.to(device)
# loading the pre-trained weights
# you have to set the path accordingly
model.load_state_dict(torch.load("part_a_best_model.pth"))

# prediction
prediction = model(images_norm)
_, prediction = torch.max(prediction, dim=1)
prediction = prediction.cpu()
# print(labels)
# print(prediction)
# print((prediction == labels).sum())

labels = labels.detach().cpu().numpy()
prediction = prediction.detach().cpu().numpy()

example =[]
for i in range(len(images_actual)):
    color = 'green' if labels[i] == prediction[i] else 'red'

    plt.figure(figsize=(4, 4))
    plt.imshow(images_actual[i])  # Assuming image is grayscale, change the cmap if needed
    plt.axis('off')
    plt.title(f'Actual: {class_labels.Class_Label[labels[i]]}\nPredicted: {class_labels.Class_Label[prediction[i]]}', fontsize=12, color=color)
    # plt.show()
    example.append(wandb.Image(plt))

# for item in example:
#     item.show()

wandb.login(key="c945d86dbf87377839ef0505ef1766e96745af82")
with wandb.init(entity="cs23m001-iit-m", project="assignment_2_plot", name="image grid"): 
    wandb.log({'Image Grid of Predicted  Test Image' : example})

wandb.finish()