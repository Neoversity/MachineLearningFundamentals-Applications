# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:59:27 2024

@author: anton
"""
import os

# Змінюємо робочу директорію на 'final_proj'
os.chdir(r"C:\Users\anton\PythonProjects\Neoversity\machine_learning_fundamentals_and_applications\machine_learning_fundamentals_and_applications\final_proj")

# Перевіряємо, чи змінилася директорія
print("Current working directory:", os.getcwd())
#%%
# Крок 1: Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Крок 2: Завантаження тренувального та валідаційного наборів даних
# Завантаження тренувального набору
train_data = pd.read_csv("final_proj_data.csv")

# Завантаження валідаційного набору
valid_data = pd.read_csv("final_proj_test.csv")

# Завантаження файл-зразок для подання прогнозів
sample_submission = pd.read_csv("final_proj_sample_submission.csv")

# Виведемо перші рядки для огляду даних
train_data.head(), valid_data.head(), sample_submission.head()
#%%
