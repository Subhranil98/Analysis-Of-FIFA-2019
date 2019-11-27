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

# Height and weight are not strongly correlated to any other variable, so they don't contribute much towards performance.
# From the above heatmap the following attributes are strongly correlated :-
# 1) Overall-Reaction : the better reaction speed of players the better will be their overall performance
# 2) Overall-Composure : More cool and calm the player better will be the performance
# 3) LongPassing,BallControl-Heading accuracy 
# 4) LongShots-ShotPower
# 5) Dribbling-Position 

# # Number of Unique Clubs and Number of players in such Clubs

# In[12]:


print("No of Unique clubs present : ",dataset['Club'].nunique())
print("No of players in each of the clubs : ", end = '\n')
print(dataset['Club'].value_counts())


# The maximum number of players is 33 in any club

# We find which clubs have 33 players in them

# In[13]:


print("Maximum No of players in any club ",max(list(dataset['Club'].value_counts())))
print(dataset['Club'].value_counts() == 33)


# These observations show that the following clubs have highest number of players:
# 
# Frosinine
# Borussia Dortmund
# Newcastle United
# Fortuna Düsseldorf
# RC Celta                      
# Cardiff City                  
# Burnley                       
# AS Monaco
# TSG 1899 Hoffenheim           
# Everton                       
# Empoli                        
# Rayo Vallecano                
# Wolverhampton Wanderers       
# Eintracht Frankfurt           
# CD Leganés
# Southampton                   
# Valencia CF

# # Finding Different Countries and No of Players in them :

# In[14]:


print("No of Unique Nationalities present : ",dataset['Nationality'].nunique())
print("No of players from each Country : ", end = '\n')
print(dataset['Nationality'].value_counts())


# As is evident from the observations , England, Germany, Spain have the highest number of players.

# # Top Player based on the following Attributes

# In[16]:


attr=['Overall','Potential','Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']


# In[17]:


for i in attr:
    print("Top performance in ",i," ",str(dataset.loc[dataset[i].idxmax()][0]))


# # Plotting graph between Overall Performance and Age

# In[18]:


unique_ages = dataset['Age'].unique()

overall_accr_ages = []

for i in unique_ages:
    d_tr_f = dataset['Age'] == i
    d = dataset[d_tr_f]
    m = d['Overall'].mean()
    overall_accr_ages.append(m)
    
plt.scatter(unique_ages,overall_accr_ages)
plt.xlabel('Unique Ages')
plt.ylabel('Overall Rating')
plt.title('Mean Overall vs Age')


# From the above plot we observe that the mean overall rating increases as the age increases upto 30. Mean Overall Rating remains constant till 35 beyond which it decreases. We infer that young players gain experience as the play over the years, then reach a saturation level, beyond which with age their performance decreases.

# # Ploting the share of each nation in term of number of players

# In[19]:


fig = plt.figure(figsize=(25, 10))
p = sns.countplot(x='Nationality', data=dataset)
_ = plt.setp(p.get_xticklabels(), rotation=90)


# # Finding distribution of Mean Overall Rating among clubs

# In[20]:


unique_clubs = list(dataset['Club'].unique())
unique_clubs = [x for x in unique_clubs if str(x) != 'nan']
top_clubs = []

overall_accr_clubs = []

for i in unique_clubs:
    d_tr_f = dataset['Club'] == i
    d = dataset[d_tr_f]
    m = d['Overall'].mean()
    if m > 75:
        top_clubs.append(i)
        overall_accr_clubs.append(m)
    

plt.bar(top_clubs, overall_accr_clubs)
# Rotation of the bars names
plt.xticks( range(len(top_clubs)), top_clubs, rotation=90)
plt.xlabel('Unique Clubs')
plt.ylabel('Overall Rating')
plt.title('Mean Overall Distribution for Top Clubs')


# As we can see , all the top 21 teams have comparable Overall Performance, though Juventis is max.

# # Observing the effect of age on Wages

# In[21]:


mean_wage_per_age = dataset.groupby('Age')['Wage'].mean()
p = sns.barplot(x = mean_wage_per_age.index, y = mean_wage_per_age.values)
p = plt.xticks(rotation=90)


# # Variation of rating with age

# In[24]:


plt.figure(figsize=[16,5])
plt.suptitle('Overall Rating Vs Age', fontsize=16)

plt.subplot(1,2,1)
bin_x = np.arange(dataset['Age'].min(), dataset['Age'].max()+1, 1)
bin_y = np.arange(dataset['Overall'].min(), dataset['Overall'].max()+2, 2)
plt.hist2d(x = dataset['Age'], y = dataset['Overall'], cmap="YlGnBu", bins=[bin_x, bin_y])
plt.colorbar()
plt.xlabel('Age (years)')
plt.ylabel('Overall Rating')

plt.subplot(1,2,2)
plt.scatter(x = dataset['Age'], y = dataset['Overall'], alpha=0.25, marker='.')
plt.xlabel('Age (years)')
plt.ylabel('Overall Rating')
plt.show()
