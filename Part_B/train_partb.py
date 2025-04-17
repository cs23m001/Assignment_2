import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import dataset
from model import Model
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def train(config):
    """
        Function to train a neural network model with specific hyper-parameter set.

        Args :
            config (dict) : collections of command-line arguments i.e hyper-parameter values.

        Return : 
            None        

    """

    #setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        print(f"The selected device is : {torch.cuda.get_device_name(device=device)}")
    else:
        print(f"The selected device is : CPU")    


    # dataset augmentatains
    transform = transforms.Compose([
            # resize to 224*224
            transforms.Resize((224, 224)),
            # convert to tensor
            transforms.ToTensor(),
            # normalize all the data between 0,1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Added extra augmentation
    aug_transforms = transforms.Compose([
            transforms.Resize((228,228)),
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


    # dataloader
    train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = config.batch_size, shuffle = True)

    # model
    model = Model(dense_size = config.dense_size, activation = config.activation, dropout = config.dropout)
    model.to(device)

    # training criterion
    criterion = nn.CrossEntropyLoss()
    # Ensuring only the classifier parameters get updataed
    optimizer = optim.Adam(model.vit_model.heads.parameters(), lr = config.learning_rate)

    # store the best model metric
    max_val_acc = 0.0

    #training loop
    for epoch in range(config.epochs):

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Iterating over the train batches
        model.train()
        for img,label in tqdm(train_loader):
            img = img.to(device = device)
            label = label.to(device = device)

            # prediction and loss compute
            prediction = model(img)
            loss = criterion(prediction, label)

            # Ensuring all the gradient of the optimizer are reset to 0
            optimizer.zero_grad()
            # gradient computation
            loss.backward()
            # parameter update
            optimizer.step()

            # for computing the accuracy and loss
            train_loss += loss.item()
            _, predicted = torch.max(prediction, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()

        train_accuracy = correct_train / total_train
        train_loss /= len(train_loader)


        # Evaluation of validation data
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        # Ensuring the no gradinet computaioin and update
        with torch.no_grad():
            for img,label in val_loader:
                img = img.to(device=device)
                label = label.to(device=device)
                # Prediction and loss computation
                prediction = model(img)
                loss = criterion(prediction, label)

                # loss and no of correct prediction accumualation
                val_loss += loss.item()
                _, predicted = torch.max(prediction, 1)
                total_val += label.size(0)
                correct_val += (predicted == label).sum().item()

            val_accuracy = correct_val / total_val
            val_loss /= len(val_loader)        

        # Saving the model based on the velidation loss
        if config.save_model:
            if max_val_acc < val_accuracy:
                print("---------------------")
                print(f"Validation accuracy {format(val_accuracy, '0.4f')}  > {format(max_val_acc, '0.4f')}. saving model parameters .....")
                print("---------------------")
                torch.save(model.state_dict(), 'part_b_fine_tune_best_model.pth')
                max_val_acc =  val_accuracy

        print(f'Epoch [{epoch+1}/{config.epochs}], ' f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    

def parse_argument():
    """
        Parses command-line arguments.

        Returns:
        argparse.Namespace : An object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Number of epochs to train neural network.", type=int, default=15)
    parser.add_argument("-b", "--batch_size", help="Batch size used to train neural network.", type=int, default=256)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate used to optimize model parameters", type=float, default=0.0003)
    parser.add_argument("-d", "--dense_size", help="Size of the fully connected layer", type=int, default=512)
    parser.add_argument("-da", "--activation", help="Activation Function for dense layer", type=str, default='relu')
    parser.add_argument("-dr", "--dropout", help="Dropout used", type=float, default=0.4)
    parser.add_argument("-aug", "--augmentation", help="Data Augmentations", type=bool, default=True)
    parser.add_argument("-save", "--save_model", help="Save the best model parameters based on validation accuracy", type=bool, default=False)

    return parser.parse_args()

if __name__ == "__main__":
    config = parse_argument()
    train(config)