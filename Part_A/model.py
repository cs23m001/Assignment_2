import torch
import torch.nn as nn

# Convolution bolck represention
# Single layer convolution layer
# Activation  followed by Maxpool
class Block(nn.Module):
    """
    A class representing a block of layers in a convolutional neural network.

    Args:
        in_channels (int): The number of input channels to the convolutional layer.
        out_channels (int): The number of output channels from the convolutional layer.
        kernel_size (int or tuple): The size of the convolutional kernel.
        activation (str): The activation function to be applied after the convolutional layer.
            Supported activation functions are 'relu', 'gelu', 'silu', and 'mish'.

    Attributes:
        conv (torch.nn.Conv2d): The convolutional layer.
        maxpool (torch.nn.MaxPool2d): The max-pooling layer.
        activation (torch.nn.Module or None): The activation function.

    """
    def __init__(self, in_channels, out_channels, kernel_size, activation): 
        super(Block, self).__init__()
        # Convolution
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = 'same')
        # Maxpool
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.activation = None

        # Selection of activation function
        if activation =='relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU() 
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            raise ValueError('Choose activation correctly ....') 
        
    # Forward pass for the convolution block
    def forward(self, inputs):
        """
            Perform a forward pass through the convolutional block.

            Args:
                inputs (torch.Tensor): The input tensor to the block.

            Returns:
                torch.Tensor: The output tensor after passing through the block.

        """
        x = self.conv(inputs)

        x = self.activation(x)            

        x = self.maxpool(x)

        return x   

class Model(nn.Module):
    """
        A convolutional neural network model.

        Args:
            filters (list): A list containing the number of filters for each convolutional layer.
            kernel_size (list): A list containing the kernel size for each convolutional layer.
            dense_size (int): The size of the dense layer. Defaults to 128.
            conv_activation (str): The activation function for convolutional layers. 
                Defaults to 'relu'.
            dense_activation (str): The activation function for the dense layer. 
                Defaults to 'relu'.
            batch_norm (bool): Whether to apply batch normalization after each convolutional block.
                Defaults to True.
            dropout (float): The dropout rate for the dropout layer. Defaults to 0.2.

    """
    def __init__(self, filters, kernel_size, dense_size = 128, conv_activation = 'relu', dense_activation = 'relu', batch_norm = True, dropout = 0.2):
        super(Model, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size 
        self.dense_size = dense_size
        self.conv_activation = conv_activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dense_activation = None

        # Defining Block and Batch Norm
        self.block1 = Block(in_channels = 3, out_channels = self.filters[0], kernel_size = self.kernel_size[0], activation = self.conv_activation)
        self.batch_norm1 = nn.BatchNorm2d(num_features = self.filters[0])

        self.block2 = Block(in_channels = self.filters[0], out_channels = self.filters[1], kernel_size = self.kernel_size[1], activation = self.conv_activation)
        self.batch_norm2 = nn.BatchNorm2d(num_features = self.filters[1])

        self.block3 = Block(in_channels = self.filters[1], out_channels = self.filters[2], kernel_size = self.kernel_size[2], activation = self.conv_activation)
        self.batch_norm3 = nn.BatchNorm2d(num_features = self.filters[2])

        self.block4 = Block(in_channels = self.filters[2], out_channels = self.filters[3], kernel_size = self.kernel_size[3], activation = self.conv_activation)
        self.batch_norm4 = nn.BatchNorm2d(num_features = self.filters[3])

        self.block5 = Block(in_channels = self.filters[3], out_channels = self.filters[4], kernel_size = self.kernel_size[4], activation = self.conv_activation)
        self.batch_norm5 = nn.BatchNorm2d(num_features = self.filters[4])

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # I assum image size 224*224. Change the in_features accordingly if the image size changes.
        self.dense = nn.Linear(in_features = (self.filters[4]*7*7), out_features = self.dense_size)
        # Output Layer for 10 classes
        self.outputs = nn.Linear(in_features = self.dense_size, out_features = 10)

        # Selection of Dense layer activation
        if dense_activation =='relu':
            self.dense_activation = nn.ReLU()
        elif self.dense_activation == 'gelu':
            self.dense_activation = nn.GELU()
        elif self.dense_activation == 'silu':
            self.dense_activation = nn.SiLU() 
        elif self.dense_activation == 'mish':
            self.dense_activation = nn.Mish()
        else:
            raise ValueError('Choose activation correctly ....') 

    def forward(self, inputs):
        """
            Perform a forward pass through the model.

            Args:
                inputs (torch.Tensor): The input tensor to the model.

            Returns:
                torch.Tensor: The output tensor of the model.

        """
        # First Block operation
        x = self.block1(inputs)
        if self.batch_norm:
            x = self.batch_norm1(x)
        
        # Second Block Operation
        x = self.block2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        
        # Third Block Operation
        x = self.block3(x)
        if self.batch_norm:
            x = self.batch_norm3(x)
        
        # Fourth Block Operation
        x = self.block4(x)  
        if self.batch_norm:
            x = self.batch_norm4(x)  
        
        # Fifth Block Operation
        x = self.block5(x)
        if self.batch_norm:
            x = self.batch_norm5(x)
        
        # Flattening operation
        # I assum the image size is 224*224. Chnage accordingly if image size changes. 
        # print(x.shape)       
        x = x.view(-1, self.filters[4]*7*7)

         # Dropout
        if self.dropout:
            x = self.dropout_layer(x)

        # Dense layer Operation
        x = self.dense(x)

        x = self.dense_activation(x)
        
        # Output
        x = self.outputs(x)

        return x