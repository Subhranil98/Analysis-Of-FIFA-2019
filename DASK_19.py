#!/usr/bin/env python
# coding: utf-8

# # Analysis of FIFA 2019

# Importing Libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import operator
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings


# Importing the Dataset

# In[7]:


dataset = pd.read_csv('F:\Engineering Books\Sem 5\Data Analytics\Data Analytics\Project\data.csv')


# In[8]:


dataset.columns


# The dataset has 89 columns, some of which are unnecessary, so we remove them.

# In[9]:


dataset = dataset.drop(["Unnamed: 0", "ID", "Photo", "Flag", "Club Logo", "Special", "Weak Foot", "Body Type", "Real Face", "Jersey Number", "Loaned From", "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW",
"LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB","Release Clause","Contract Valid Until","Joined"], axis = 1)

print("Number of columns remaining is ",len(dataset.columns))

dataset_columns = list(dataset.columns)


# Next we find how many missing values are there in each of the attributes and clean the dataset thereafter.

# In[10]:


l = []
c = 0

for i in dataset_columns:
    if(dataset[i].isnull().any()):
        print(i," ",sum(dataset[i].isnull().values.ravel()))
        if(sum(dataset[i].isnull().values.ravel()) < 100):
            l.append(c)
    c += 1
            
dataset = dataset.dropna()


dataset['Value'] = dataset['Value'].str[1:-1]
dataset["Value"] = pd.to_numeric(dataset["Value"])
dataset['Wage'] = dataset['Wage'].str[1:-1]
dataset["Wage"] = pd.to_numeric(dataset["Wage"])
dataset['Height'] = dataset['Height'].str[:1]+'.'+dataset['Height'].str[3:]
dataset["Height"] = pd.to_numeric(dataset["Height"])
dataset['Weight'] = dataset['Weight'].str[:-3]
dataset["Weight"] = pd.to_numeric(dataset["Weight"])

dataset.isnull().values.any()
dataset = dataset.dropna()
dataset.isnull().values.any()


# Now our dataset does not have any missing values.
# Next we plot a HeatMap to find Correlations between the Attributes

# In[11]:


def plotCorrelationMatrix(df, graphWidth):
    #filename = df.dataframeName
    #df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    #plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


plotCorrelationMatrix(dataset, 16)
