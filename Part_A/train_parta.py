import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import dataset
from model import Model
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def train(config):
    """
        This function trains a model based on specific hyper-parameter settings.
        
        Args : 
            config (dict) : command line argument, collection of hyper-parameters for training

        Returns:
            None    
    
    """

    #setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print(f"The selected device is : {torch.cuda.get_device_name(device=device)}")
    else:
        print(f"The selected device is CPU")    

    # dataset augmentatains
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # Augmented trasformation
    aug_transforms = transforms.Compose([
            # Resize to 228*228
            transforms.Resize((228,228)),
            # Random Horizontal Flip
            transforms.RandomHorizontalFlip(),
            # Random Rotaion by 30 degree
            transforms.RandomRotation(degrees=10),
            # Randomccrop  and resize to 224*224
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            # Transform to tensor
            transforms.ToTensor(),
            # Normalization
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
        
        # iterating over the tain loader
        model.train()
        for img,label in tqdm(train_loader):
            # Transfering tp gpu
            img = img.to(device = device)
            label = label.to(device = device)

            # forward pass through model
            prediction = model(img)
            # Loss calculation
            loss = criterion(prediction, label)
            
            # Ensuring gradient are set to 0
            optimizer.zero_grad()
            # Gradient computaion
            loss.backward()
            # Udpate the parameters
            optimizer.step()

            # Total loss accumulation
            train_loss += loss.item()
            # Prediction class index .ie same as np.argmax()
            _, predicted = torch.max(prediction, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()

        # Accuracy and loss for single epoch
        train_accuracy = correct_train / total_train
        train_loss /= len(train_loader)


        # Evaluation on validation data
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        # Ensuring no gradient calculation
        with torch.no_grad():
            for img,label in val_loader:
                img = img.to(device=device)
                label = label.to(device=device)
                # Forward pass through model
                prediction = model(img)
                # val loss computaion per batch
                loss = criterion(prediction, label)

                # Accumulation of batch wise loss
                val_loss += loss.item()
                # Predeicted class index
                _, predicted = torch.max(prediction, 1)
                total_val += label.size(0)
                correct_val += (predicted == label).sum().item()

            # Validaton accuracy and loss
            val_accuracy = correct_val / total_val
            val_loss /= len(val_loader)        

        # Saving the model based on best validation accuracy
        if config.save_model:
            if max_val_acc < val_accuracy:
                print("---------------------")
                print(f"Validation accuracy {format(val_accuracy, '0.4f')}  > {format(max_val_acc, '0.4f')}. saving model parameters .....")
                print("---------------------")
                torch.save(model.state_dict(), 'part_a_best_model.pth')
                max_val_acc =  val_accuracy

        print(f'Epoch [{epoch+1}/{config.epochs}], ' f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    
        


def parse_argument():
    parser = argparse.ArgumentParser(description="Arguments for training the model")

    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.00035, help="Learning rate for training")
    parser.add_argument("--filters", nargs='+', type=int, default=[32, 64, 64, 128, 256], help="List of filter sizes for convolutional layers")
    parser.add_argument("--kernel_size", nargs='+', type=int, default=[3, 3, 3, 3, 3], help="List of kernel sizes for convolutional layers")
    parser.add_argument("--dense_size", type=int, default=128, help="Size of dense layer")
    parser.add_argument("--conv_activation", type=str, default='mish', help="Activation function for convolutional layers")
    parser.add_argument("--dense_activation", type=str, default='silu', help="Activation function for dense layer")
    parser.add_argument("--batch_norm", type=bool, default=True, help="Whether to use batch normalization")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--augmentation", type=bool, default=True, help="Whether to use data augmentation")
    parser.add_argument("--save_model", type=bool, default=False, help="Whether to save the trained model")

    return parser.parse_args()

if __name__ == "__main__":
    config = parse_argument()
    train(config)
