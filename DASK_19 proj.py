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


dataset['Value'] = dataset['Value'].str[1:-1]
dataset["Value"] = pd.to_numeric(dataset["Value"])
dataset['Wage'] = dataset['Wage'].str[1:-1]
dataset["Wage"] = pd.to_numeric(dataset["Wage"])
dataset['Height'] = dataset['Height'].str[:1]+'.'+dataset['Height'].str[3:]
dataset["Height"] = pd.to_numeric(dataset["Height"])
dataset['Weight'] = dataset['Weight'].str[:-3]
dataset["Weight"] = pd.to_numeric(dataset["Weight"])


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

'''
Height and weight are not strongly correlated to any other variable, so they don't contribute much towards performance.
 From the above correlogram the following attributes are strongly coorelated :-
1) Overall-Reaction : the better reaction speed of players the better will be their overall performance
2) Overall-Composure : More cool and calm the player better will be the performance
3) LongPassing,BallControl-Heading accuracy 
4) LongShots-ShotPower
5) Dribbling-Position 
'''

plt.plot(dataset['Age'],dataset['Potential']) 
  
# naming the x axis 
plt.xlabel('Age') 
# naming the y axis 
plt.ylabel('Potential') 
  
# giving a title to my graph 
plt.title('Relation potential and age with respected value of players') 
  
# function to show the plot 
plt.show() 

#No of unique clubs and no. of players in each such club

print("No of Unique clubs present : ",dataset['Club'].nunique())
print("No of players in each of the clubs : ", end = '\n')
print(dataset['Club'].value_counts())

#We aim to find the club which has maximum no. of players

print("Maximum No of players in any club ",max(list(dataset['Club'].value_counts())))
print(dataset['Club'].value_counts() == 33)

#These observations show that the following clubs have highest number of players
'''
Frosinine
Borussia Dortmund
Newcastle United
Fortuna Düsseldorf
RC Celta                      
Cardiff City                  
Burnley                       
AS Monaco
TSG 1899 Hoffenheim           
Everton                       
Empoli                        
Rayo Vallecano                
Wolverhampton Wanderers       
Eintracht Frankfurt           
CD Leganés
Southampton                   
Valencia CF       

'''

#Finding Different Countries and No of Players in them

print("No of Unique Nationalities present : ",dataset['Nationality'].nunique())
print("No of players from each Country : ", end = '\n')
print(dataset['Nationality'].value_counts())

#As is evident from the observations , England, Germany, Spain have the highest number of players.

#We aim to find Top Player based on the following Attributes

attr=['Overall','Potential','Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

for i in attr:
    print("Top performance in ",i," ",str(dataset.loc[dataset[i].idxmax()][0]))
    
    
#Plotting graph between Overall Performance and Age
    
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

#From the above plot we observe that the mean overall rating increases as the age increases upto 30.
# Mean Overall Rating remains constant till 35 beyond which it decreases.
#We infer that young players gain experience as the play over the years, then reach a saturation level,
#beyond which with age their performance decreases.
