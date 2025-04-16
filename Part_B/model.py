import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Model(nn.Module):
  """
      Neural network model class representation.
      
      Attributes :
            dense_size : Number of neurons in the first linear layer of the classifier
            num_class : Number of the output classes.
            activation : Activation function used for the first linear layer in the classifer.
            dropout : Precentage of dropout used between the two linear layer of the classifier. 
            vit_model : The pre-trained vision transformer model

  """
  def __init__(self, dense_size, activation = 'relu', dropout = 0.0):
    super(Model, self).__init__()
    self. desne_size = dense_size
    self.num_classes = 10
    self.activation = None
    self.dropout = nn.Dropout(p=dropout)
    
    # Selection of activation function
    if activation == 'relu':
      self.activation = nn.ReLU()
    elif activation == 'gelu':
      self.activation = nn.GELU() 
    elif activation == 'silu':
      self.activation = nn.SiLU()
    else:
      raise ValueError("Choose your actvation correctly .....")   
      
    # Loading the pre-trained weights
    self.vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # Output size of the last linear layer
    self.in_features = self.vit_model.heads.head.in_features

    # Freezing all the layers
    for params in self.vit_model.parameters():
      params.requires_grad = False

    # Difining the new classifier and adding it with the vit_model heads
    self.vit_model.heads = nn.Sequential(
        nn.Linear(self.in_features, dense_size),
        self.activation,
        self.dropout,
        nn.Linear(dense_size, self.num_classes)
        )

  # Forward pass 
  def forward(self, img):
    x = self.vit_model(img)
    return x