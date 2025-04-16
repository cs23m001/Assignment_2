import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import dataset
from model import Model

def evaluate(config):
    """
        Evaluate a trained model on a test dataset.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Must contain the key 'model_path' which specifies the path to the pre-trained model.

        Returns:
            None

    """
    # Defining the transformations
    transform = transforms.Compose([
                # Resize to 224*224
                transforms.Resize((224, 224)),
                # Transforming to tensor
                transforms.ToTensor(),
                # Normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    test_dataset = dataset.iNaturalist(img_path =  os.path.join(os.getcwd(),'utils','test_data.csv'), transforms = transform)
    test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = True)

    # Defineing the model structure
    model = Model(filters = [32, 64, 64, 128, 256], kernel_size = [3, 3, 3, 3, 3], dense_size = 128, batch_norm = True, dropout = 0.1)
    model.to(device)
    # loading the pre-trained weights
    model.load_state_dict(torch.load(config.model_path))
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # evaluating the model 
    model.eval()
    test_loss = 0.0
    total_correct = 0
    total_eval_sample = 0

    with torch.no_grad():
        # Iterate through test samples
        for img,label in tqdm(test_loader):
            img = img.to(device=device)
            label = label.to(device=device)
            # Prediction
            prediction = model(img)
            # Loss calculation
            loss = criterion(prediction, label)
            test_loss += loss.item()
            # Predected class index
            _, prediction = torch.max(prediction, dim = 1)
            total_correct += (prediction == label).sum().item()
            total_eval_sample += label.size(0)
        # Accuracy and Loss calculation
        test_accuracy = total_correct / total_eval_sample
        test_loss = test_loss / len(test_loader)

        print(f"Test Accuracy is : {test_accuracy : 0.4f}")





def parse_argument():
    """
        Parses command-line arguments.

        Returns:
        argparse.Namespace : An object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", help="Path to the Model directory", type=str, default='part_a_best_model.pth')
    
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_argument()
    evaluate(config)