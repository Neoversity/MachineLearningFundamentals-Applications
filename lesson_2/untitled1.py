# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:14:55 2024

@author: anton
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Завантаження даних з правильним кодуванням
data = pd.read_csv('SeoulBikeData.csv', parse_dates=[0], encoding='latin1')

# Перетворення колонки Date в datetime з dayfirst=True
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Перевірка перших рядків
print(data.head())


# Агрегація даних по днях
data_daily = data.groupby(data['Date']).agg({
    'Rented Bike Count': 'sum',
    'Hour': 'last',
    'Temperature(°C)': 'mean',
    'Humidity(%)': 'mean',
    'Wind speed (m/s)': 'mean',
    'Visibility (10m)': 'mean',
    'Dew point temperature(°C)': 'mean',
    'Solar Radiation (MJ/m2)': 'mean',
    'Rainfall(mm)': 'sum',
    'Snowfall (cm)': 'sum',
    'Seasons': 'last',
    'Holiday': 'last',
    'Functioning Day': 'last'
}).reset_index().drop('Hour', axis=1)

# Перегляд перших 5 рядків агрегованих даних
data_daily.head()

# Побудова парних графіків для візуалізації даних
sns.pairplot(data_daily)

# Побудова теплової карти кореляцій
sns.heatmap(data_daily.corr(numeric_only=True), center=0, cmap='coolwarm', annot=True)

# Перегляд перших 5 значень дня
data_daily.Date.dt.day.head()

# %%
data_daily.drop()
