import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# This function collect all the image path and labels from the given directory
def collect_path(dir_path):
    img_path, labels = [], []
    for idx,files in enumerate(sorted(os.listdir(dir_path))):
        for img in os.listdir(os.path.join(dir_path, files)):
            path = os.path.join(dir_path, files, img)
            img_path.append(path)
            labels.append(idx)
    return img_path, labels        

# This function split the train set into train data and validation data.
# After split it save the files as .csv file
def split_traindata(val_split = 0.2):
    metadata = pd.read_csv('train_meta_data.csv')
    # Split data into train and validation sets
    # Stratify mode ensures the class balancing split
    train_data, val_data = train_test_split(metadata, test_size=val_split, stratify=metadata['label'])

    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)

# This function takes the path and process the data by calling previous functions
def process_data(config):
    train_path, train_labels = collect_path(config.train_dir)
    data={'path' : train_path, 'label' : train_labels}
    df = pd.DataFrame(data)
    df.to_csv('train_meta_data.csv', index=False)

    test_path, test_labels = collect_path(config.test_dir)
    data={'path' : test_path, 'label' : test_labels}
    df = pd.DataFrame(data)
    df.to_csv('test_data.csv', index=False)

    split_traindata(val_split = config.val_split)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train_dir", help="Path to the train data directory", type=str, default='/home/apu/Desktop/nature_12K/inaturalist_12K/train')
    parser.add_argument("-test", "--test_dir", help="Path to the test data directory", type=str, default='/home/apu/Desktop/nature_12K/inaturalist_12K/val')
    parser.add_argument("-split", "--val_split", help="Train and validation split", type=float, default=0.2)

    return parser.parse_args()

if __name__ == "__main__":
    config = parse_argument()
    process_data(config)


