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
#Перевірка структури набору даних
train_data.info(), train_data.describe(include='all')
#%%
# Пропущені значення кількість
missing_counts = train_data.isnull().sum()
missing_percentage = (missing_counts / len(train_data)) * 100

# Аналіз числових та категоріальних колонок окремо
numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Summary числові 
numerical_missing = missing_percentage[numerical_cols]
numerical_summary = train_data[numerical_cols].describe()

# Summary категориальні 
categorical_missing = missing_percentage[categorical_cols]
categorical_summary = train_data[categorical_cols].nunique()

# відсутнє для візуалізації
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentage
}).sort_values(by='Missing Percentage', ascending=False)

#%%
# шоу пропушені значення
import matplotlib.pyplot as plt



# Plot missing values
plt.figure(figsize=(12, 6))
missing_summary['Missing Percentage'].head(50).plot(kind='bar', title='Top 50 Columns with Missing Data', xlabel='Columns')
plt.ylabel('Missing Percentage (%)')
plt.show()

numerical_missing, categorical_missing, categorical_summary
#%%
# Крок 1: Видалення колонок із понад 90% пропусків
missing_percentage = (train_data.isnull().sum() / len(train_data)) * 100
columns_to_drop = missing_percentage[missing_percentage > 90].index
train_data_cleaned = train_data.drop(columns=columns_to_drop)
#%%
# Крок 2: Заповнення пропусків
# Числові колонки
numerical_cols = train_data_cleaned.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if train_data_cleaned[col].isnull().sum() > 0:
        train_data_cleaned[col] = train_data_cleaned[col].fillna(train_data_cleaned[col].mean())

# Категоріальні колонки
categorical_cols = train_data_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if train_data_cleaned[col].isnull().sum() > 0:
        train_data_cleaned[col] = train_data_cleaned[col].fillna(train_data_cleaned[col].mode()[0])
#%%
# Крок 3: Кодування змінних
# Target Encoding для високої кількості унікальних значень
large_unique_cols = [col for col in categorical_cols if train_data_cleaned[col].nunique() > 10]
encoder = TargetEncoder(cols=large_unique_cols)
train_data_encoded = encoder.fit_transform(train_data_cleaned, train_data_cleaned['y'])

# OneHot Encoding для низької кількості унікальних значень
small_unique_cols = [col for col in categorical_cols if train_data_cleaned[col].nunique() <= 10]
train_data_encoded = pd.get_dummies(train_data_encoded, columns=small_unique_cols, drop_first=True)

# Перевірка на залишкові пропуски
remaining_missing = train_data_encoded.isnull().sum()
print("Пропуски після обробки:")
print(remaining_missing[remaining_missing > 0])

# Збереження оброблених даних для подальшого аналізу
train_data_encoded.to_csv('processed_train_data.csv', index=False)
#%%
# Відобразити перші кілька рядків
print(train_data_encoded.head())
# перевірити основну статистику
print(train_data_encoded.describe())