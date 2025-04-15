import os
import pandas as pd


Class_Label = {
    0 : 'Amphibia',
    1 : 'Animalia',
    2 : 'Arachnida',
    3 : 'Aves',
    4 : 'Fungi',
    5 : 'Insecta',
    6 : 'Mammalia',
    7 : 'Mollusca',
    8 : 'Plantae',
    9 : 'Reptilia'
}

# Reading corresponding files
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

print("----------------------------------------------------------")
print("The number training samples per class : ")
print("----------------------------------------------------------")
for label in range(10):
    count = 0
    for i in range (len(train_data)):
        if train_data.iloc[i , 1] == label:
            count += 1
    print(f"The number of samples for class {Class_Label[label]} is : {count}")  

print()
print() 

print("----------------------------------------------------------")
print("The number validation samples per class : ")
print("----------------------------------------------------------")
for label in range(10):
    count = 0
    for i in range (len(val_data)):
        if val_data.iloc[i , 1] == label:
            count += 1
    print(f"The number of samples for class {Class_Label[label]} is : {count}")  

print()
print()

print("----------------------------------------------------------")
print("The number test samples per class : ")
print("----------------------------------------------------------")
for label in range(10):
    count = 0
    for i in range (len(test_data)):
        if test_data.iloc[i , 1] == label:
            count += 1
    print(f"The number of samples for class {Class_Label[label]} is : {count}")  