# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:59:27 2024

@author: anton
"""
import os

# Змінюємо робочу директорію на 'hw_8'
os.chdir(r"C:\Users\anton\PythonProjects\Neoversity\machine_learning_fundamentals_and_applications\machine_learning_fundamentals_and_applications\hw_8")

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
train_data = pd.read_csv("mod_04_hw_train_data.csv")

# Завантаження валідаційного набору
valid_data = pd.read_csv("mod_04_hw_valid_data.csv")

# Виведемо перші рядки для огляду даних
train_data.head(), valid_data.head()
#%%
# Крок 3: Первинний аналіз даних (EDA) для перевірки пропусків
# Перевіримо наявність пропусків у тренувальному та валідаційному наборах даних
missing_train = train_data.isnull().sum()
missing_valid = valid_data.isnull().sum()

#%%
missing_data_analysis = pd.DataFrame({
    "Train Missing Before": missing_train,
    "Train Missing After": train_data.isnull().sum(),
    "Valid Missing Before": missing_valid,
    "Valid Missing After": valid_data.isnull().sum()
})

# Виведемо результати аналізу пропусків у текстовому форматі
print("Аналіз пропусків у тренувальному наборі даних:")
print(missing_data_analysis[["Train Missing Before", "Train Missing After"]])

print("\nАналіз пропусків у валідаційному наборі даних:")
print(missing_data_analysis[["Valid Missing Before", "Valid Missing After"]])
#%%
from datetime import datetime

# Видалимо стовпці 'Name' та 'Phone_Number'
train_data.drop(columns=['Name', 'Phone_Number'], inplace=True)
valid_data.drop(columns=['Name', 'Phone_Number'], inplace=True)

# Перетворимо 'Date_Of_Birth' у вік (кількість років)
current_year = datetime.now().year

train_data['Age'] = train_data['Date_Of_Birth'].apply(lambda x: current_year - int(x.split('/')[-1]))
valid_data['Age'] = valid_data['Date_Of_Birth'].apply(lambda x: current_year - int(x.split('/')[-1]))

# Видалимо стовпець 'Date_Of_Birth', оскільки ми вже отримали вік
train_data.drop(columns=['Date_Of_Birth'], inplace=True)
valid_data.drop(columns=['Date_Of_Birth'], inplace=True)

# Перевіримо, як тепер виглядають набори даних
train_data.head(), valid_data.head()
#%%
# Заповнимо пропуски у тренувальному наборі
train_data.fillna({
    'Experience': train_data['Experience'].mean(),
    'Qualification': train_data['Qualification'].mode()[0],
    'Role': train_data['Role'].mode()[0],
    'Cert': train_data['Cert'].mode()[0]
}, inplace=True)

# Перевіримо, чи залишилися пропуски
missing_train_final = train_data.isnull().sum()
missing_train_final
#%%
# Крок 4: Масштабування числових ознак

# Категоризація змінної 'Experience'
def categorize_experience(years):
    if years < 3:
        return 'Junior'
    elif 3 <= years <= 7:
        return 'Mid'
    else:
        return 'Senior'

# Додаємо категорію до тренувального та валідаційного наборів
train_data['Experience_Category'] = train_data['Experience'].apply(categorize_experience)
valid_data['Experience_Category'] = valid_data['Experience'].apply(categorize_experience)

# Видаляємо числову ознаку 'Experience'
train_data.drop(columns=['Experience'], inplace=True)
valid_data.drop(columns=['Experience'], inplace=True)

# Масштабування лише числових змінних
numerical_features = ['Age']  # Лише 'Age', оскільки 'Experience' тепер є категоріальною

# Ініціалізація трансформерів
scaler_standard = StandardScaler()
scaler_power = PowerTransformer()

# Копіюємо дані для масштабування
train_data_standard = train_data.copy()
valid_data_standard = valid_data.copy()

train_data_power = train_data.copy()
valid_data_power = valid_data.copy()

# Масштабування за допомогою StandardScaler
train_data_standard[numerical_features] = scaler_standard.fit_transform(train_data[numerical_features])
valid_data_standard[numerical_features] = scaler_standard.transform(valid_data[numerical_features])

# Масштабування за допомогою PowerTransformer
train_data_power[numerical_features] = scaler_power.fit_transform(train_data[numerical_features])
valid_data_power[numerical_features] = scaler_power.transform(valid_data[numerical_features])

# Перевіримо результати
train_data_standard.head(), train_data_power.head()
#%%
# Ініціалізація TargetEncoder
encoder = TargetEncoder()

# Визначення категоріальних і числових змінних
categorical_features = ['Qualification', 'University', 'Role', 'Cert', 'Experience_Category']
numerical_features = ['Age']

# Копіюємо дані для масштабування
train_data_encoded = train_data.copy()
valid_data_encoded = valid_data.copy()

# Кодуємо категоріальні змінні TargetEncoding
for col in categorical_features:
    train_data_encoded[col] = encoder.fit_transform(train_data[col], train_data['Salary'])
    valid_data_encoded[col] = encoder.transform(valid_data[col])

# Об'єднуємо числові та закодовані ознаки
all_features = numerical_features + categorical_features

# Масштабування всіх ознак
scaler_standard = StandardScaler()
scaler_power = PowerTransformer()
#%%
# Масштабування для StandardScaler
train_data_standard = train_data_encoded.copy()
valid_data_standard = valid_data_encoded.copy()

train_data_standard[all_features] = scaler_standard.fit_transform(train_data_encoded[all_features])
valid_data_standard[all_features] = scaler_standard.transform(valid_data_encoded[all_features])

# Масштабування для PowerTransformer
train_data_power = train_data_encoded.copy()
valid_data_power = valid_data_encoded.copy()

train_data_power[all_features] = scaler_power.fit_transform(train_data_encoded[all_features])
valid_data_power[all_features] = scaler_power.transform(valid_data_encoded[all_features])
#%%
# Розділення даних
X_train_standard = train_data_standard.drop(columns=['Salary'])
y_train_standard = train_data_standard['Salary']

X_valid_standard = valid_data_standard.drop(columns=['Salary'])
y_valid_standard = valid_data_standard['Salary']

X_train_power = train_data_power.drop(columns=['Salary'])
y_train_power = train_data_power['Salary']

X_valid_power = valid_data_power.drop(columns=['Salary'])
y_valid_power = valid_data_power['Salary']
#%%
# Оптимізація n_neighbors для KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error


knn = KNeighborsRegressor()
param_grid = {'n_neighbors': range(1, 50)}
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=5, n_jobs=-1)

X_train = train_data_standard.drop(columns=['Salary'])
y_train = train_data_standard['Salary']

grid_search.fit(X_train, y_train)

best_n_neighbors = grid_search.best_params_['n_neighbors']
best_score = -grid_search.best_score_

print(f"Найкраще значення n_neighbors: {best_n_neighbors}")
print(f"MAPE для найкращого значення n_neighbors: {best_score:.2%}")

# Крок 5: Навчання моделі з оптимальним n_neighbors
best_knn_model = KNeighborsRegressor(n_neighbors=best_n_neighbors)
best_knn_model.fit(X_train, y_train)

# Підготовка валідаційного набору
X_valid = valid_data_standard.drop(columns=['Salary'])
y_valid = valid_data_standard['Salary']

# Продовження виконання моделі
y_pred = best_knn_model.predict(X_valid)
final_mape = mean_absolute_percentage_error(y_valid, y_pred)
final_mse = mean_squared_error(y_valid, y_pred)
final_r2 = r2_score(y_valid, y_pred)

print(f"Validation Results with Optimized n_neighbors:")
print(f" - Validation MAPE: {final_mape:.2%}")
print(f" - Validation MSE: {final_mse:.2f}")
print(f" - Validation R2: {final_r2:.2f}")

#%%
print(X_train.isnull().sum())
print(y_train.isnull().sum())
