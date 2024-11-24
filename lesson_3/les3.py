# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:32:43 2024

@author: anton
"""

#%%

import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
#%%
data = pd.read_csv('../datasets/mod_03_topic_05_weather_data.csv.gz')
data.shape
#%%
data.dtypes
#%%

data.isna().mean().sort_values(ascending=False)
#%%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tmp = (data
           .groupby('Location')
           .apply(lambda x:
                  x.drop(['Location', 'Date'], axis=1)
                  .isna()
                  .mean()))

plt.figure(figsize=(9, 13))

ax = sns.heatmap(tmp,
                 cmap='Blues',
                 linewidth=0.5,
                 square=True,
                 cbar_kws=dict(
                     location="bottom",
                     pad=0.01,
                     shrink=0.25))

ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=90)
#%%
data = data[data.columns[data.isna().mean().lt(0.35)]]

data = data.dropna(subset='RainTomorrow')
#%%

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')
#%%
melted = data_num.melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=4,
                  sharex=False,
                  sharey=False,
                  aspect=1.25)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()
#%%

ConfusionMatrixDisplay.from_predictions(y_test, pred)