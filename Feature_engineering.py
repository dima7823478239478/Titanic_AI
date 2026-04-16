import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sys

train_data_csv_path = ("/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean.csv")
data = pd.read_csv(train_data_csv_path)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
data['IsChild'] = (data['Age'] < 12).astype(int)
data['WomanWithChild'] = ((data['Sex'] == 'female') & (data['Parch'] > 0)).astype(int)
data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)

data.to_csv('/Users/dmitrii/PycharmProjects/Titanic_AI_project/data_clean_fe.csv', index=False)