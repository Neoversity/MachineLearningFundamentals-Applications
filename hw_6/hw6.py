# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:43:37 2024

@author: anton
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#%%
df = pd.read_csv('rain.csv')
df.dropna(inplace=True)