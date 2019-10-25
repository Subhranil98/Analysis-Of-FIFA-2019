import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('F:\Engineering Books\Sem 5\Data Analytics\Data Analytics\Project\data.csv')

print(dataset.head(10))

dataset.columns

dataset = dataset.drop(["Unnamed: 0", "ID", "Photo", "Flag", "Club Logo", "Special", "Weak Foot", "Skill Moves", "Body Type", "Real Face", "Jersey Number", "Loaned From", "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW",
"LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB"], axis = 1)

print(len(dataset.columns))

dataset_columns = list(dataset.columns)

l = []
c = 0

for i in dataset_columns:
    if(dataset[i].isnull().any()):
        print(i," ",sum(dataset[i].isnull().values.ravel()))
        if(sum(dataset[i].isnull().values.ravel()) < 100):
            l.append(c)
    c += 1
            
dataset = dataset.drop(l)
