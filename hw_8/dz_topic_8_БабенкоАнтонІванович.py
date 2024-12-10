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
# Ініціалізація трансформерів
scaler_standard = StandardScaler()
scaler_power = PowerTransformer()

# Вибір числових змінних для трансформації
numerical_features = ['Experience', 'Age']

# Масштабування за допомогою StandardScaler
train_data_standard = train_data.copy()
valid_data_standard = valid_data.copy()

train_data_standard[numerical_features] = scaler_standard.fit_transform(train_data[numerical_features])
valid_data_standard[numerical_features] = scaler_standard.transform(valid_data[numerical_features])

# Масштабування за допомогою PowerTransformer
train_data_power = train_data.copy()
valid_data_power = valid_data.copy()

train_data_power[numerical_features] = scaler_power.fit_transform(train_data[numerical_features])
valid_data_power[numerical_features] = scaler_power.transform(valid_data[numerical_features])

# Перевіримо результати трансформацій
train_data_standard.head(), train_data_power.head()
#%%
# Ініціалізація OneHotEncoder для кодування категоріальних змінних
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop='first' для уникнення мультиколінеарності
categorical_features = ['Qualification', 'University', 'Role', 'Cert']

# Кодування для StandardScaler трансформованих даних
encoded_train_standard = encoder.fit_transform(train_data_standard[categorical_features])
encoded_train_standard_df = pd.DataFrame(
    encoded_train_standard, 
    columns=encoder.get_feature_names_out(categorical_features)
)
train_data_standard_encoded = pd.concat(
    [train_data_standard.drop(columns=categorical_features), encoded_train_standard_df], axis=1
)

encoded_valid_standard = encoder.transform(valid_data_standard[categorical_features])
encoded_valid_standard_df = pd.DataFrame(
    encoded_valid_standard, 
    columns=encoder.get_feature_names_out(categorical_features)
)
valid_data_standard_encoded = pd.concat(
    [valid_data_standard.drop(columns=categorical_features), encoded_valid_standard_df], axis=1
)

# Кодування для PowerTransformer трансформованих даних
encoded_train_power = encoder.fit_transform(train_data_power[categorical_features])
encoded_train_power_df = pd.DataFrame(
    encoded_train_power, 
    columns=encoder.get_feature_names_out(categorical_features)
)
train_data_power_encoded = pd.concat(
    [train_data_power.drop(columns=categorical_features), encoded_train_power_df], axis=1
)

encoded_valid_power = encoder.transform(valid_data_power[categorical_features])
encoded_valid_power_df = pd.DataFrame(
    encoded_valid_power, 
    columns=encoder.get_feature_names_out(categorical_features)
)
valid_data_power_encoded = pd.concat(
    [valid_data_power.drop(columns=categorical_features), encoded_valid_power_df], axis=1
)

# Перевіримо, як виглядають дані після кодування
train_data_standard_encoded.head(), train_data_power_encoded.head()
#%%
# Крок 5: Побудова моделі за допомогою KNeighborsRegressor


# Розділення даних на ознаки (features) та цільову змінну (target)
X_train_standard = train_data_standard_encoded.drop(columns=['Salary'])
y_train_standard = train_data_standard_encoded['Salary']

X_valid_standard = valid_data_standard_encoded.drop(columns=['Salary'])
y_valid_standard = valid_data_standard_encoded['Salary']

# Побудова моделі KNeighborsRegressor для StandardScaler даних
knn_model_standard = KNeighborsRegressor(n_neighbors=5)
knn_model_standard.fit(X_train_standard, y_train_standard)

# Прогноз для валідаційного набору
y_pred_standard = knn_model_standard.predict(X_valid_standard)
#%%
# Крок 6: Оцінка точності моделі (MAPE)
from sklearn.metrics import mean_absolute_percentage_error
# Обчислення MAPE
mape_standard = mean_absolute_percentage_error(y_valid_standard, y_pred_standard)

# Аналогічно для PowerTransformer
X_train_power = train_data_power_encoded.drop(columns=['Salary'])
y_train_power = train_data_power_encoded['Salary']

X_valid_power = valid_data_power_encoded.drop(columns=['Salary'])
y_valid_power = valid_data_power_encoded['Salary']

knn_model_power = KNeighborsRegressor(n_neighbors=5)
knn_model_power.fit(X_train_power, y_train_power)

y_pred_power = knn_model_power.predict(X_valid_power)
mape_power = mean_absolute_percentage_error(y_valid_power, y_pred_power)

# Виведемо результати
print(f"Validation MAPE (StandardScaler): {mape_standard:.2%}")
print(f"Validation MAPE (PowerTransformer): {mape_power:.2%}")
#%%
# Крок 7: Розрахунок додаткових метрик точності (MSE, R2)
# Розрахуємо метрики точності регресійної моделі для StandardScaler

# Mean Squared Error (MSE) для StandardScaler
mse_standard = mean_squared_error(y_valid_standard, y_pred_standard)

# R-squared (R2) для StandardScaler
r2_standard = r2_score(y_valid_standard, y_pred_standard)

# Розрахуємо метрики для PowerTransformer
mse_power = mean_squared_error(y_valid_power, y_pred_power)
r2_power = r2_score(y_valid_power, y_pred_power)

# Виведемо результати
print(f"StandardScaler Results:")
print(f" - Validation MAPE: {mape_standard:.2%}")
print(f" - Validation MSE: {mse_standard:.2f}")
print(f" - Validation R2: {r2_standard:.2f}")

print(f"\nPowerTransformer Results:")
print(f" - Validation MAPE: {mape_power:.2%}")
print(f" - Validation MSE: {mse_power:.2f}")
print(f" - Validation R2: {r2_power:.2f}")