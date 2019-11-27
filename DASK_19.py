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

# # Finding Similar Players using KNN - Recommendation System

# In[25]:


#Selecting columns to find similarity among players

attributes = dataset.iloc[:, 16:]
attributes['Skill Moves'] = dataset['Skill Moves']
attributes['Age'] = dataset['Age']
workrate = dataset['Work Rate'].str.get_dummies(sep='/ ')
attributes = pd.concat([attributes, workrate], axis=1)
df = attributes
attributes = attributes.dropna()
df['Name'] = dataset['Name']
df['Position'] = dataset['Position']
df = df.dropna()
print(attributes.columns)

#Displaying our attribute set
attributes.head()

#Standardize the dataset

scaled = StandardScaler()
X = scaled.fit_transform(attributes)

#Create recommendations using NearestNeighbors ML

recommendations = NearestNeighbors(n_neighbors=5,algorithm='kd_tree')
recommendations.fit(X)

#Get Similar Players index

player_index = recommendations.kneighbors(X)[1]

player_index

#Defining a recommend function to display results

def get_index(x):
    return df[df['Name']==x].index.tolist()[0]

def recommend_similar(player):
    print("These are 4 players similar to {} : ".format(player))
    index=  get_index(player)
    for i in player_index[index][1:]:
        print("Name: {0}\nPosition: {1}\n".format(df.iloc[i]['Name'],df.iloc[i]['Position']))


# Recommending Similar Players for Lionel Messi

# In[26]:


recommend_similar('L. Messi')


# Recommending Similar Players for Christiano Ronaldo

# In[27]:


recommend_similar('Cristiano Ronaldo')

# # Wage Prediction

# In[30]:


X = dataset.iloc[:,[1,3,4,6,9]]
y = dataset['Wage']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=101)

'''
from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()
Xtrain = stsc.fit_transform(Xtrain)
Xtest = stsc.fit_transform(Xtest)

'''

def pred_wage(degree, Xtrain, Xtest, ytrain):
    if degree > 1:
        poly = PolynomialFeatures(degree = degree)
        Xtrain = poly.fit_transform(Xtrain)
        Xtest = poly.fit_transform(Xtest)
    lm = LinearRegression()
    lm.fit(Xtrain, ytrain)
    wages = lm.predict(Xtest)
    return wages

predicted_wages1 = pred_wage(3, Xtrain, Xtest, ytrain)

sns.regplot(ytest, predicted_wages1, scatter_kws={'alpha':0.3, 'color':'y'})
plt.xlabel('Actual Wage')
plt.ylabel('Predicted Wage')
plt.show()

predicted_wages2 = pred_wage(4, Xtrain, Xtest, ytrain)

sns.regplot(ytest, predicted_wages2, scatter_kws={'alpha':0.3, 'color':'y'})
plt.xlabel('Actual Wage')
plt.ylabel('Predicted Wage')
plt.show()

sns.distplot(ytest-predicted_wages1, bins=200, hist_kws={'color':'r'}, kde_kws={'color':'y'})
plt.xlim(-50, 50)


# As we can see that the residual wages are centered at 0, so our model is good enough

#  Lionel Messi has Age = 31, Overall = 94, Potential = 94, Value = 110, Int. Rep = 5

# In[31]:


y_testing = pred_wage(4,Xtrain,[[31,94,94,110,5]],ytrain)


# In[33]:


print("Predicted Wage is ", y_testing[0])


# His predicted Wage is 527.37 Million and his actual Wage is 565 Million

# Neymar Junior has Age = 26, Overall = 92, Potential = 93, Value = 118, Int. Rep = 5

# In[34]:


y_testing_2 = pred_wage(4,Xtrain,[[26,92,93,118,5]],ytrain)
print("Predicted Wage is ", y_testing_2[0])


# His predicted Wage is 293.4 Million and his actual Wage is 290 Million


# # Recommending a Team based on Feature Values to Club Managers 

# In[35]:


player_features = ['Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
       'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
       'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
       'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
       'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']

df_postion  = pd.DataFrame()
for position_name, features in dataset.groupby(dataset['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    df_postion[position_name] = tuple(top_features)
df_postion.head()


df_best = pd.DataFrame.copy(dataset)
df_position_y = pd.DataFrame.copy(df_postion)
del df_position_y['RAM']
df_best.head()
posi = []
player = []
club_l = []
for col in df_position_y.columns:
    tmp_df = pd.DataFrame()
    #print(col)
    l = [df_postion[col].values]
    l = l[0]
    l = list(l)
    l.append('Name')
    tmp_df = pd.DataFrame.copy(df_best[df_best['Position'] == col][l])
    tmp_df['mean'] = np.mean(tmp_df.iloc[: , :-1] , axis = 1)
    name = tmp_df['Name'][tmp_df['mean'] == tmp_df['mean'].max()].values[0]
    club = df_best['Club'][df_best['Name'] == str(name)].values[0]
    
    posi.append(col)
    player.append(name)
    club_l.append(club)
    
gk = ['GK']
forward = ['LS', 'ST', 'RS','LF', 'CF', 'RF']
midfeilder = ['LW','RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
              'RCM', 'RM', 'LDM', 'CDM', 'RDM' ]
defenders = ['LWB','RWB', 'LB', 'LCB', 'CB',]
    
print()
print('GoalKeeper : ')
for p , n , c in zip(posi , player , club_l):
    if p in gk:
        print('{} [Club : {} , Position : {} , Age : {}]'.format(n , c , p ,
                                                                dataset['Age'][dataset['Name'] == n].values[0]))

print('\nFORWARD : ')
for p , n , c in zip(posi , player , club_l):
    if p in forward:
        print('{} [Club : {} , Position : {} , Age : {}]'.format(n , c , p , 
                                                                dataset['Age'][dataset['Name'] == n].values[0]))
print('\nMIDFEILDER : ')
for p , n , c in zip(posi , player , club_l):
    if p in midfeilder:
        print('{} [Club : {} , Position : {} , Age : {}]'.format(n , c , p , 
                                                                dataset['Age'][dataset['Name'] == n].values[0]))
print('\nDefender : ')
for p , n , c in zip(posi , player , club_l):
    if p in defenders:
        print('{} [Club : {} , Position : {} , Age : {}]'.format(n , c , p , 
                                                                dataset['Age'][dataset['Name'] == n].values[0]))
        
# # Clustering Players based on K-Means Clustering and Elbow Method:

# # Performing K Means Clustering on Players Based on Age , Overall and Potential

# In[36]:


X = dataset.iloc[:,[1,3,4]].values


# Finding the Ideal Number of Clusters using Elbow Method

# In[37]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #kmeans.inertia_ finds Sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[38]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
# x-coordinate = X[y_kmeans == 0,0] 2nd 0 for the 0th column , y_kmeans==0 for Cluster 1
# y-coordinate = X[y_kmeans == 0,1]
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
'''
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# # Performing K Means Clustering on Players Based on their Playing Features

# In[39]:


X = dataset.iloc[:,15:].values


# 
# Using the elbow method to find the optimal number of clusters

# In[40]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #kmeans.inertia_ finds Sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



  
