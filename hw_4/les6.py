# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:06:51 2024

@author: anton
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from prosphera.projector import Projector
#%%
data = pd.read_csv('datasets/mod_03_topic_06_diabets_data.csv')
data.head()
#%%
data.info()
#%%
X, y = (data.drop('Outcome', axis=1), data['Outcome'])

cols = ['Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI']

X[cols] = X[cols].replace(0, np.nan)
#%%

ax = sns.scatterplot(x=X['Glucose'], y=X['BMI'], hue=y)
ax.vlines(x=[120, 160],
          ymin=0,
          ymax=X['BMI'].max(),
          color='black',
          linewidth=0.75)
#%%
