# Part B: Fine-tuning a pre-trained model
In this part, a pre-trained network is fine-tuned for image classification. I have used the vosion_transformer from torchvision.models for the fine-tuning. The mentioned model is loaded with the default i.e `ViT_B_16_Weights` which is trained on the `ImageNet`. You can also download the pre-trained weights oof the vision transformer from [here](https://download.pytorch.org/models/vit_b_16-c867db91.pth). Wandb is used for finding the sitable hyper-parameters for training. The model is trained and evaluated using the [iNaturalist]() dataset.

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
The training and evaluation code requires above mentioned packages. Note that the code has only been tested with the specified versions and also expects a `Linux environment with GPU(>=4 gb for training)`. To setup all the required dependencies for training and evaluation, please follow the instructions below:

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
    pip install -r requirements.txt
```

* Download or Clone git repository
```bash
git clone https://github.com/cs23m001/Assignment_2.git
cd Assignment_2
cd Part_B
```

# Repository Deatils
The repository contains the list of python files. 
* utils 
  * `class_labels.py` - Contains a dictionaty mapping all the classes.
  * `csv_generator.py` - This file will generate 4 .cs filev containing the image file path and corresponding labels. All the .csv files will used to create the Dataset class for the DataLoader.
  * `dataset.py` - This file contains the class definition of the dataset class used in this assignment. 
  * `count_split_samples.py` - Report the number of samples per class in each of the training, validation and test split.

* `model.py` - The vision transormer model. Contains two part. features_extractor and the classifier.
* `train_partb.py` - Python script to train the model based on specific settings.
* `evaluate_partb.py` - Contains python script for evaluation of the model on the test data.
* `wandb_params_search.py` - Contains code for hyper-parameter search using wandb.
* `requirements.txt` - Contains additional packages for installation.

# How To Run

**During running you might find some hidden file. Due to which you might get some error. If you encounter such error due to hidden file(in my case i found some .DS file inside  train and train/Fungi directory), first press crtl+h to to see the hidden file and then delete it.** <br>
After setting the environment and cloning the repository, first run the `csv_generator.py` file inside the `utils` directory. 
```python
python csv_generator.py --train_dir=<path to train dir> --test_dir=<path to test dir> --val_split=0.2
```
For example `train_dir = '/home/.../.../nature_12K/inaturalist_12K/train'`

This will create 4 .csv files
* `train_meta_data.csv` - Containing all the image path and labels inside the training directory.
* `train_data.csv` - Containing all the image path and labels for the training data.
* `val_data.csv` - Containing all the image path and labels for the validation data.
* `test_data.csv` - Containing all the image path and labels for the testing data.

If you want to check for numbers of samples per class for each of the slipt just run
```bash
python count_split_samples.py
```
The .csv files are used to define the Dataset class. Once all the csv file are generated you can train the model.


# Training

To train the model run the `train_partb.py` script with the desired command-line arguments. 
```bash
python train_partb.py --epochs=15 --batch_size=256 --learning_rate=0.0003 --dense_size512 --activation='relu' --dropout=0.4 --augmentation=True --save_model=False
```

`If you use save_model=True make sure you have set the path to correctly. You need to change path in the section of code where saving is done. While loading the model use the same path only.`

This command will optimize a classification model that is based on a vision transformer. With a batch size of 256 and a learning rate of 0.0003, the model will be trained over 15 epochs. The final dense layer of the model will include 512 neurons with relu activation. Additionally, there will be a 0.4 percent dropout used between the classifier's output layer and its final dense layer. Only the classifier module will be trained.

## List of command line arguments

* `--epochs` - Number of epochs to train neural network.
* `--batch_size` - Batch size used to train neural network.
* `--learning_rate` - Learning rate used to optimize model parameters.
* `--dense_size` - Size of the fully connected layer of the classifier.
* `--activation` - Activation Function for dense layer.
* `--dropout` - Dropout used in between the last dense layer and output layer.
* `--augmentation` - Data Augmentations. Boolean variable.
* `--save_model` - Save the best model parameters based on validation accuracy.

If you want to train the model with the default parameters then just run
```python
    python train_partb.py
```

or you can run it for 3 epochs
```bash
python train_partb.py --epochs=3 --batch_size=256 --learning_rate=0.0003 --dense_size=512 --activation='relu' --dropout=0.4 --augmentation=True --save_model=False
```

The default configuration is set based on the wandb hyper-parameter search.

# Evaluation 

Before evaluation make sure you have created the 4 .csv files by running the `csv_generator.py` inside `utils` directory. To evaluate the model first download the pre-trained weights from [here](https://drive.google.com/file/d/1eX1dRHXnBB3SdWmANOxfltACG27YaawL/view?usp=drive_link) inside the same directory and then run

```bash
python evaluate_partb.py
```

This will first initialize the model with the default hyper-parameters used during training. Finally it will test the model on the seperate test data in the `iNaturalist` dataset and report the test accuracy.
`All the reported accuracies are based on the specific split set`

# Acknowledgements

* [http://www.cse.iitm.ac.in/~miteshk/CS6910.html](http://www.cse.iitm.ac.in/~miteshk/CS6910.html)
* [The Math Behind Fine-Tuning Deep Neural Networks](https://towardsdatascience.com/the-math-behind-fine-tuning-deep-neural-networks-8138d548da69)
* [VisionTransformer - PyTorch](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)

