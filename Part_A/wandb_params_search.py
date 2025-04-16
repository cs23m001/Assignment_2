import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
import pandas as pd 
from utils import dataset
from model import Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
import wandb


def train(config=None):
    
    with wandb.init(config=config):
        
        config = wandb.config
        wandb.run.name = wandb.run.name = 'eph_' + str(config.epochs) + '-bs_' + str(
            config.batch_size) + '-dense_' + str(config.dense_size) + '-lr_' + str(
                config.learning_rate) + '-dr_' + str(config.dropout)  

        #collect data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"The selected device is : {torch.cuda.get_device_name(device=device)}")

        # dataset augmentatains
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        aug_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset definition
        train_data = dataset.iNaturalist(img_path =  os.path.join(os.getcwd(),'utils','train_data.csv'), transforms = aug_transforms if config.augmentation else transform)
        val_data = dataset.iNaturalist(img_path =  os.path.join(os.getcwd(),'utils','val_data.csv'), transforms = transform)

        print(f"Number of Training samples : {len(train_data)}")
        print(f"Number of Valdidation samples : {len(val_data)}")


        #dataloader
        train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True)
        val_loader = DataLoader(val_data, batch_size = config.batch_size, shuffle = True)

        #model
        model = Model(filters = config.filters, kernel_size = config.kernel_size, dense_size = config.dense_size, batch_norm = config.batch_norm, dropout = config.dropout)

        # Multi-gpu setup
        if torch.cuda.device_count() > 1:
            print("Multi-gpu Traing started ...")
            model = nn.DataParallel(model)
            
        model.to(device)

        #training criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
        # print(model)

        # store the best model metric
        max_val_acc = 0.0

        #training loop
        for epoch in range(config.epochs):

            train_loss = 0.0
            correct_train = 0
            total_train = 0

            model.train()
            for img,label in tqdm(train_loader):
                img = img.to(device = device)
                label = label.to(device = device)

                prediction = model(img)
                loss = criterion(prediction, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(prediction, 1)
                total_train += label.size(0)
                correct_train += (predicted == label).sum().item()

            train_accuracy = correct_train / total_train
            train_loss /= len(train_loader)



            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for img,label in tqdm(val_loader):
                    img = img.to(device=device)
                    label = label.to(device=device)
                    prediction = model(img)
                    loss = criterion(prediction, label)

                    val_loss += loss.item()
                    _, predicted = torch.max(prediction, 1)
                    total_val += label.size(0)
                    correct_val += (predicted == label).sum().item()

                val_accuracy = correct_val / total_val
                val_loss /= len(val_loader)        


            # if config.save_model == True:
            #     if max_val_acc < val_accuracy:
            #         print("---------------------")
            #         print(f"Validation accuracy {format(val_accuracy, '0.4f')}  > {format(max_val_acc, '0.4f')}. saving model parameters .....")
            #         print("---------------------")
            #         torch.save(model.state_dict(), '/home/apu/Desktop/Best Model/Part A/best_model')
            #         max_val_acc =  val_accuracy

            print(f'Epoch [{epoch+1}/{config.epochs}], ' f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')  

            wandb.log( {"training_accuracy" : train_accuracy, "training_loss" : train_loss,
                        "validation_accuracy" : val_accuracy, "validation_loss" : val_loss, "epoch" : epoch} )     
            
        wandb.run.save()  

    del model
    torch.cuda.empty_cache()



if __name__ == "__main__":
    metric = {
                'name' : 'validation_accuracy',
                'goal' : 'maximize'
            }
    parameters = {
                    'epochs' : { 'values' : [10, 15] },
                    'batch_size' : { 'values' : [32, 64, 128, 256] },
                    'learning_rate' : { 'values' : [0.01, 0.001, 0.0003, 0.00035] },     
                    'filters' : { 'values' : [
                        [32, 32, 32, 32, 32], [32, 32, 64, 64, 64], [64, 64, 64, 64, 64], [64, 64, 128, 128, 128], [32, 64, 64, 128, 128], [32, 64, 64, 128, 256]
                    ] },
                    'kernel_size' : { 'values' : [
                       [3, 3, 3, 3, 3], [3, 3, 3, 5, 5], [3, 3, 5, 5, 7], [3, 3, 5, 7, 7]
                    ] },
                    'dense_size' : { 'values' : [128, 256, 512] },
                    'conv_activation' : { 'values' : ['relu', 'gelu', 'silu', 'mish'] },
                    'dense_activation' : { 'values' : ['relu', 'gelu', 'silu', 'mish'] },  
                    'batch_norm' : { 'values' : [True, False] }, 
                    'dropout' : { 'values' : [0, 0.1, 0.2, 0.35, 0.4, 0.45] }, 
                    'augmentation' : { 'values' : [True, False] },                                       

                } 
    
    sweep_config = dict()
    sweep_config['name'] = 'run_2'
    sweep_config['method'] = 'bayes'
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters
    #config = parse_argument()
    wandb.login(key="xxxxxxxxxx")
    sweep_id = wandb.sweep(sweep_config, entity="xxxx", project="assignment_2")
    wandb.agent(sweep_id, function=train, count=60)
    wandb.finish()  