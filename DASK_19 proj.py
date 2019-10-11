import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


fifa_19 = pd.read_csv('F:/Sem 5/Elective II-Data Analytics/Project/fifa19/data.csv')

print(fifa_19.head(10))

fifa_19 = fifa_19.drop(["Unnamed: 0", "ID", "Photo", "Flag", "Potential", "Club Logo", "Wage", "Special", "Preferred Foot", "International Reputation", "Weak Foot", "Skill Moves", "Work Rate", "Body Type", "Real Face", "Jersey Number", "Joined", "Loaned From", "Contract Valid Until", "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW",
"LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", "Release Clause"], axis = 1)

fifa_19.drop(labels = [2571, 2572, 2573, 2574], inplace = True)
fifa_19.set_index([pd.Index(range(fifa_19.shape[0]))], inplace = True)
fifa_19.isnull().sum()

plt.figure(figsize=(50,40))
p = sns.heatmap(fifa_19.corr(), annot=True)
print(p)