# Assignment 2 : Learn how to use CNNs: train from scratch and finetune a pre-trained model as it is.
The primary objective of this assignment was to create a Convolution Neural Network (CNN) for the purpose of classifying images. The assignment is completed in two parts. A five-layer CNN model is designed and trained entirely from scratch in the first part. The second part involves fine-tuning the previously trained model, specifically the `vision trainsformer` trained on `ImageNet`, for the same task.

# Wandb Report Link
Please find the `wandb` report for this assignment [here](https://wandb.ai/apusarkar2195/Assignment2_CNN/reports/CS6910-Assignment-2--Vmlldzo3Mzk1NDM2).

# Dataset
For this assignment [iNaturalist]() is used for training and evaluation. The directory strucutre of the dataset 
```bash
nature_12k/
    ├── inaturalist12k/
         ├── train/
         │   ├── Amphibia/
         │   │   ├── image1.jpg
         │   │   ├── image2.jpg
         │   │   └── ...
         │   ├── Animalia/
         │   │   ├── image1.jpg
         │   │   ├── image2.jpg
         │   │   └── ...
         │   └── ...
         ├── val/
         │   ├── Amphibia/
         │   │   ├── image1.jpg
         │   │   ├── image2.jpg
         │   │   └── ...
         │   ├── Animalia/
         │   │   ├── image1.jpg
         │   │   ├── image2.jpg
         │   │   └── ...
         │   └── ...
```

There are total `10 classes`. Each class inside train directory contains `1000 samples` and inside val directory contains `200 samples`. For this assignment, the validaion set is created using `0.2` percent of the train samples and samples inside the `val` directory is used for testing.
The total numners of samples 
* training - 8000 (800 per calss)
* validation - 2000 (200 per class)
* test - 2000 (200 per class)

# General Pipieline
To train and fine-tune the [iNaturalist]() dataset used. The pipeline for the assignment is 
* Data Preparation
* Model Design
* Training
* Evaluation

