# Part A: Training from scratch
Here, I have design a 5 layer Convolution Neural Network and then train the network from scratch. Wandb is used for finding the sitable hyper-parameters for training. The model is trained and evaluated using the [iNaturalist]() dataset.

# Wandb Report Link

Please find the `wandb` report link for this assignment [here](https://wandb.ai/cs23m001-iit-m/assignment_2_B/reports/DA6401-Assignment-2--VmlldzoxMjI5MzAzMA). 

# Library Used
The list of libraries used are : 
* `PyTorch` - Model building, training, various utility task like tranformation, dataloader etc.
* `PIL` - Image reading and manipulation.
* `Pandas` - Dataframe createion and reading
* `OS` - Path manipulation.
* `scikit-learn` - For computation and data split.
* `matplotlib,tqdm` - Visualization.
* `Numpy` - Handling mathematical operations.
* `argparse` - Handling command line arguments.
* `wandb` - Finding best hyper-parameter for training.

# Installation
The training and evaluation code requires above mentioned packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

* Create conda environment
```bash
conda create --name assignment_2
conda activate assignment_2
```
* Install pytorch
Then install pytorch using the forllowing command 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
`All the code in this experiment are trained and tested using gpu enabel system. Hence to run all the code properly a gpu enable machine is must.` If you find any dificulties then install `pytorch` version and corresponding `cuda` version accordingly.

* Installing extra requirements
The above command will download most of the necessary dependencies along with pytorch. For installing the rest of the packages run 
``` bash
    pip install requirements.txt
```

* Download or Clone git repository
```bash
git clone https://github.com/cs23m001/Assignment_2.git
cd Assignment_2
cd Part_A
```

# Repository Deatils
The repository contains the list of python files. 
* utils 
  * `class_labels.py` - Contains a dictionaty mapping all the classes.
  * `csv_generator.py` - This file will generate 4 .csv filev containing the image file path and corresponding labels. All the .csv files will used to create the Dataset class for the DataLoader.
  * `dataset.py` - This file contains the class definition of the dataset class used in this assignment. 
  * `count_split_samples.py` - Report the number of samples per class in each of the training, validation and test split.

* `model.py` - This file contains the 5 layered CNN model defination.
* `train_parta.py` - Python script to train the model based on specific settings.
* `evaluate_parta.py` - Contains python script for evaluation of the model on the test data.
* `image_grid.py` - It will create 10*3 image gird on the test samples.
* `wandb_gridd_push.py` - To push the image grid to wandb.
* `wandb_params_search.py` - Contains code for hyper-parameter search using wandb.
* `requirements.txt` - Contains additional packages for installation.
* `part_a_best_model.pth` - Pre-trained weights for the Part A.

`If you want to visualize the image grid make sure you replace the weights path inside the code properly.`

# How To Run

**During running you might find some hidden file. Due to which you might get some error. If you encounter such error due to hidden file(in my case i found some .DS file inside  train and train/Fungi directory), first press crtl+h to to see the hidden file and then delete it.** <br>
After setting the environment and cloning the repository, first run the `csv_generator.py` file inside the `utils` directory. 
```python
python csv_generator.py --train_dir=<path to train dir> --test_dir=<path to test dir> --val_split=0.2
```
`For example the train_dir = '/home/.../.../nature_12K/inaturalist_12K/train'` 

This will create 4 .csv files
* `train_meta_data.csv` - Containing all the image path and labels inside the training directory.
* `train_data.csv` - Containing all the image path and labels for the training data.
* `val_data.csv` - Containing all the image path and labels for the validation data.
* `test_data.csv` - Containing all the image path and labels for the testing data.

If you want to check for numbers of samples per class for each of the slipt just run
```python
python count_split_samples.py
```
The .csv files are used to define the Dataset class. Once all the csv file are generated you can train the model.


# Training
To train the model run the `train_parta.py` script with the desired command-line arguments. 

```bash
python train_parta.py --epochs=15 --batch_size=128 --learning_rate=0.0003 --filters 32 64 64 128 128 --kernel_size 3 3 3 5 5 --dense_size=512 --conv_activation='gelu' --dense_activation='mish' --batch_norm=True --dropout=0.45 --augmentation=True --save_model=False

```
This command will train a five-layer CNN model with kernel sizes of 3, 3, 3, 5, and 5 with 32, 64, 64, 128 and 128 filters in each consecutive layer. Each of these convolution layers will have gelu as activation function. Specifically, 512 neurons with 'mish' activation will be used in the last fully connected layer prior to the output layer. With data augmentation, the model will train for 15 epochs with a batch size of 128 and a learning rate of 0.0003. Batch normalization and dropout with probability 0.45 will also be used in the training procedure for regularization.

## List of command line arguments
* `--epochs` - Number of epochs to train neural network.
* `--batch_size` - atch size used to train neural network.
* `--learning_rate` - Learning rate used to optimize model parameters.
* `--filters` - Number of filters per CNN block.
* `--kernel_size` - Kernel size for each CNN blick.
* `--dense_size` - Size of the fully connected layer.
* `--conv_activation` - Activation Function for Convolution layer.
* `--dense_activation` - Activation Function for dense layer.
* `--batch_norm` - Batch Normalization. Boolean variable.
* `--dropout` - Dropout used in between the last dense layer and output layer.
* `--augmentation` - Data Augmentations. Boolean variable.
* `--save_model` - Save the best model parameters based on validation accuracy.

If you want to train the model with the default parameters then just run
```python
    python train_parta.py
```

or you can run this for 3 epochs
```bash
  python train_parta.py --epochs=3 --batch_size=128 --learning_rate=0.0003 --filters 32 64 64 128 128 --kernel_size 3 3 3 5 5 --dense_size=512 --conv_activation='relu' --dense_activation='gelu' --batch_norm=True --dropout=0.4 --augmentation=True --save_model=False
```    

The default configuration is set based on the wandb hyper-parameter search. It will take approximately `3 min per epoch (on a system with NVIDIA GeForce GTX TITAN X(12GB) gpu)`.

# Evaluation 
Before evaluation make sure you have created the 4 .csv files by running the `csv_generator.py` inside `utils` directory.
In order to evaluate the model make sure you have set the `model_path` as pre-trained path and run

```bash
python evaluate_parta.py
```
As the repo already contained the pre-trained weights you just need to run the above command to evaluate the model.

This will first initialize the model with the default hyper-parameters used during training and then will load the pre-trained weights. Finally it will test the model on the seperate test data in the `iNaturalist` dataset and report the test accuracy.
`All the repoted accuracies are tested on the given split of the tset, train and validation`

# Acknowledgements

* [http://www.cse.iitm.ac.in/~miteshk/CS6910.html](http://www.cse.iitm.ac.in/~miteshk/CS6910.html)
* [https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
* [https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/)
